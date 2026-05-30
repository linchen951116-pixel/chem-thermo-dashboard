import streamlit as st
import pubchempy as pcp
import py3Dmol
import networkx as nx
import plotly.graph_objects as go
import time
import pandas as pd
import numpy as np
from deep_translator import GoogleTranslator
import re
import requests
import streamlit.components.v1 as components

# 網頁基礎設定 (必須在最第一行)
st.set_page_config(page_title="中文化學物質分析與動態熱力學系統", layout="wide")

st.title("🧪 物質深度分析 & 3D 動態熱力學系統")
st.markdown("內建 **Plotly 原生硬體加速動畫引擎**，徹底告別畫面閃爍與破圖，享受 60FPS 的極致流暢熱能擴散體驗。")

# ==========================================
# 核心一：維基百科學術名詞對接引擎
# ==========================================
LOCAL_CHEM_DICT = {
    "阿斯匹靈": "Aspirin", "普拿疼": "Acetaminophen", "雙氧水": "Hydrogen peroxide",
    "鹽酸": "Hydrochloric acid", "硫酸": "Sulfuric acid", "硝酸": "Nitric acid",
    "氨水": "Ammonia", "食鹽": "Sodium chloride", "氯化鈉": "Sodium chloride",
    "硝酸鉀": "Potassium nitrate", "亞硝酸鉀": "Potassium nitrite",
    "KNO2": "Potassium nitrite", "kno2": "Potassium nitrite",
    "高錳酸鉀": "Potassium permanganate", "碳酸鈉": "Sodium carbonate",
    "氫氧化鈉": "Sodium hydroxide", "乙醇": "Ethanol", "甲醇": "Methanol",
    "苯": "Benzene", "水": "Water"
}

def contains_chinese(text):
    return bool(re.search('[\u4e00-\u9fff]', text))

def translate_via_wikipedia(zh_name):
    try:
        url = f"https://zh.wikipedia.org/w/api.php?action=query&prop=langlinks&titles={zh_name}&lllang=en&format=json"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36'}
        res = requests.get(url, headers=headers, timeout=5).json()
        pages = res.get("query", {}).get("pages", {})
        for page_id, page_info in pages.items():
            if "langlinks" in page_info: return page_info["langlinks"][0]["*"] 
    except: pass
    return None

def fix_chemical_formula(formula):
    if not formula: return formula
    formula_fix_map = {
        "ClNa": "NaCl", "HNaO": "NaOH", "ClK": "KCl", "HKO": "KOH",
        "IK": "KI", "KNO2": "KNO₂", "NO2K": "KNO₂", "NO3K": "KNO₃",
        "C2H4O2": "CH₃COOH", 
    }
    if formula in formula_fix_map: return formula_fix_map[formula]
    for metal in ["Na", "K", "Ca", "Mg", "Al", "Fe", "Cu", "Zn", "Ag"]:
        if metal in formula and not formula.startswith(metal):
            if not formula.startswith("C"): 
                formula = metal + formula.replace(metal, "")
    return formula

# ==========================================
# 核心二：自動 3D/2D 結構檔爬蟲 & 智慧物態萃取引擎
# ==========================================
def safe_translate(text):
    if not text or text == "無相關文獻數據": return text
    try: return GoogleTranslator(source='en', target='zh-TW').translate(text)
    except: return text

def simplify_physical_state(text):
    if not text or text == "無相關文獻數據": return text
    text_lower = text.lower()
    is_solution = "solution" in text_lower or "aqueous" in text_lower
    is_solid = any(kw in text_lower for kw in ["solid", "crystal", "powder", "pellet", "crystalline", "granule", "chunk", "salt"])
    is_gas = any(kw in text_lower for kw in ["gas", "vapor"])
    is_liquid = any(kw in text_lower for kw in ["liquid", "fluid"])
    
    if is_solid: return "🧊 固體"
    elif is_liquid and not is_solution: return "💧 液體"
    elif is_gas: return "☁️ 氣體"
    elif is_solution: return "💧 水溶液 (純物質通常為固體)"
    else: return text

def get_mol_sdf(cid):
    try:
        url_3d = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/record/SDF/?record_type=3d"
        res3d = requests.get(url_3d, timeout=5)
        if res3d.status_code == 200: return res3d.text, "3D 立體"
    except: pass
    return None, "2D 平面"

def fetch_sds_and_properties(cid):
    props = {
        "外觀與性狀": "無相關文獻數據", "密度": "無相關文獻數據", "熔點": "無相關文獻數據",
        "沸點": "無相關文獻數據", "閃點": "無相關文獻數據", "溶解度": "無相關文獻數據",
        "蒸氣壓": "無相關文獻數據", "危險信號詞": "無標示 / 安全", "危害警告": []
    }
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
        res = requests.get(url, timeout=6).json()
        sections = res.get("Record", {}).get("Section", [])
        for sec in sections:
            if sec.get("TOCHeading") == "Chemical and Physical Properties":
                for subsec in sec.get("Section", []):
                    if subsec.get("TOCHeading") == "Experimental Properties":
                        for prop in subsec.get("Section", []):
                            heading = prop.get("TOCHeading")
                            try:
                                val = prop["Information"][0]["Value"]["StringWithMarkup"][0]["String"]
                                if heading == "Physical Description": props["外觀與性狀"] = simplify_physical_state(val)
                                elif heading == "Density": props["密度"] = val if "g/" in val or "kg/" in val else f"{val} g/cm³"
                                elif heading == "Melting Point": props["熔點"] = val
                                elif heading == "Boiling Point": props["沸點"] = val
                                elif heading == "Flash Point": props["閃點"] = val
                                elif heading == "Solubility": props["溶解度"] = val
                                elif heading == "Vapor Pressure": props["蒸氣壓"] = val
                            except: continue
            elif sec.get("TOCHeading") == "Safety and Hazards":
                for subsec in sec.get("Section", []):
                    if subsec.get("TOCHeading") == "Hazards Identification":
                        for ghs in subsec.get("Section", []):
                            if ghs.get("TOCHeading") == "GHS Classification":
                                for info in ghs.get("Information", []):
                                    if info.get("Name") == "Signal":
                                        raw_signal = info["Value"]["StringWithMarkup"][0]["String"]
                                        if "Danger" in raw_signal: props["危險信號詞"] = "危險 (Danger)"
                                        elif "Warning" in raw_signal: props["危險信號詞"] = "警告 (Warning)"
                                        else: props["危險信號詞"] = raw_signal
                                    elif info.get("Name") == "GHS Hazard Statements":
                                        raw_hazards = [h["String"] for h in info["Value"]["StringWithMarkup"]]
                                        props["危害警告"] = [safe_translate(h) for h in raw_hazards[:5]]
    except: pass
    return props

def generate_crystal_lattice_html(elements, style):
    viewer = py3Dmol.view(width=450, height=350)
    viewer.setBackgroundColor('#f0f2f6')
    color_map = {"Na": "purple", "K": "violet", "Cl": "green", "O": "red", "N": "blue", "Default": "orange"}
    el1 = elements[0] if len(elements) > 0 else "Default"
    el2 = elements[1] if len(elements) > 1 else "O"
    c1, c2 = color_map.get(el1, color_map["Default"]), color_map.get(el2, color_map["Default"])
    
    for x in [-2, 0, 2]:
        for y in [-2, 0, 2]:
            for z in [-2, 0, 2]:
                if (x + y + z) % 4 == 0:
                    current_color, r = c1, 0.9 if style == "sphere" else 0.5
                else:
                    current_color, r = c2, 1.2 if style == "sphere" else 0.6
                viewer.addSphere({'center': {'x': x*3, 'y': y*3, 'z': z*3}, 'radius': r, 'color': current_color})
                
    if style in ["stick", "sphere", "line"]:
        for i in [-2, 0, 2]:
            for j in [-2, 0, 2]:
                viewer.addLine({'start': {'x': -6, 'y': i*3, 'z': j*3}, 'end': {'x': 6, 'y': i*3, 'z': j*3}, 'color': 'gray'})
                viewer.addLine({'start': {'x': i*3, 'y': -6, 'z': j*3}, 'end': {'x': i*3, 'y': 6, 'z': j*3}, 'color': 'gray'})
                viewer.addLine({'start': {'x': i*3, 'y': j*3, 'z': -6}, 'end': {'x': i*3, 'y': j*3, 'z': 6}, 'color': 'gray'})
                
    viewer.zoomTo()
    return viewer._make_html().replace("http://", "https://")

# --- 側邊欄：全局參數設定面板 ---
with st.sidebar:
    st.header("⚙️ 全局參數設定面板")
    st.subheader("🔬 1. 物質百科檢索")
    user_input = st.text_input("輸入化學式、中文試劑或藥品名稱", "Sodium chloride").strip()
    style = st.selectbox("3D 顯示風格", ["stick", "sphere", "line", "cross"])
    search_button = st.button("🔍 執行數據檢索", type="primary")
    
    st.markdown("---")
    st.subheader("🔥 2. 熱傳導動態模擬參數")
    env_temp = st.slider("環境溫度設定 (°C)", min_value=-20.0, max_value=60.0, value=25.0, step=0.5)
    init_temp = st.slider("中心粒子點火溫度 (°C)", min_value=50.0, max_value=500.0, value=300.0, step=10.0)
    k_val = st.slider("熱傳導係數 (k)", min_value=0.01, max_value=0.50, value=0.15, step=0.01)
    # 動畫速度改為毫秒級距，越小越快
    anim_speed = st.slider("動畫播放速度 (每幀毫秒)", min_value=10, max_value=200, value=50, step=10)

# ==========================================
# 狀態機與資料快取初始化
# ==========================================
if 'mol_atoms' not in st.session_state:
    st.session_state.mol_atoms = list(range(10))
    st.session_state.mol_bonds = [(0,4), (0,5), (0,6), (1,4), (1,7), (1,8), (2,5), (2,7), (2,9), (3,6), (3,8), (3,9)]
    st.session_state.core_node = 0
    st.session_state.edge_node = 9
    st.session_state.mol_name = "預設測試結構"

if 'particle_temps' not in st.session_state:
    st.session_state.particle_temps = {i: env_temp for i in st.session_state.mol_atoms}
    st.session_state.particle_temps[0] = init_temp

# 搜尋行為攔截與快取
if search_button and user_input:
    with st.spinner("🧠 系統正在調閱學術詞典並重建分子拓樸..."):
        try:
            english_name = user_input
            if contains_chinese(user_input) or user_input in LOCAL_CHEM_DICT:
                if user_input in LOCAL_CHEM_DICT:
                    english_name = LOCAL_CHEM_DICT[user_input]
                else:
                    wiki_name = translate_via_wikipedia(user_input)
                    if wiki_name: english_name = wiki_name
                    else:
                        translated = GoogleTranslator(source='auto', target='en').translate(user_input)
                        if contains_chinese(translated):
                            st.warning(f"⚠️ 無法自動辨識「{user_input}」，請嘗試輸入常見俗名。")
                            st.stop()
                        else: english_name = translated

            compounds = pcp.get_compounds(english_name, 'name')
            if compounds:
                c = compounds[0]
                atoms = [atom.aid for atom in c.atoms]
                bonds = [(bond.aid1, bond.aid2) for bond in c.bonds]
                
                degree = {}
                for a, b in bonds:
                    degree[a] = degree.get(a, 0) + 1
                    degree[b] = degree.get(b, 0) + 1
                c_node = max(degree, key=degree.get) if degree else (atoms[0] if atoms else 0)
                e_node = min(degree, key=degree.get) if degree else (atoms[-1] if atoms else 0)
                
                sds_data = fetch_sds_and_properties(c.cid)
                fixed_formula = fix_chemical_formula(c.molecular_formula)
                sdf_data, dim_type = get_mol_sdf(c.cid)
                
                st.session_state.search_data = {
                    "english_name": english_name.capitalize(), "cid": c.cid, "fixed_formula": fixed_formula,
                    "molecular_weight": c.molecular_weight, "tpsa": c.tpsa, "h_bond_donor_count": c.h_bond_donor_count,
                    "h_bond_acceptor_count": c.h_bond_acceptor_count, "isomeric_smiles": c.isomeric_smiles,
                    "sds_data": sds_data, "sdf_data": sdf_data, "dim_type": dim_type,
                    "unique_elements": list(set([atom.element for atom in c.atoms]))
                }

                st.session_state.mol_atoms = atoms
                st.session_state.mol_bonds = bonds
                st.session_state.core_node = c_node
                st.session_state.edge_node = e_node
                st.session_state.mol_name = st.session_state.search_data["english_name"]
                
                # 重置溫度狀態
                st.session_state.particle_temps = {i: env_temp for i in atoms}
                st.session_state.particle_temps[c_node] = init_temp
            else:
                st.error(f"⚠️ 找不到與「{english_name}」匹配的物質。")
        except Exception as e:
            st.error(f"檢索發生錯誤：{e}")

# --- 雙分頁介面 ---
tab1, tab2 = st.tabs(["🧬 SDS 物質安全與化學百科", "🔥 極速動畫渲染台 (無閃爍)"])

# ==========================================
# 分頁 1：化學百科與 SDS 危害報告 (基於快取渲染)
# ==========================================
with tab1:
    if 'search_data' in st.session_state:
        sd = st.session_state.search_data
        st.success(f"✅ 檢索成功！物質映射：「**{sd['english_name']}**」 | 真實原子數: {len(st.session_state.mol_atoms)}")
        
        col_left, col_right = st.columns([1, 1.3])
        with col_left:
            st.subheader("⚛️ 空間立體結構")
            if sd["dim_type"] == "3D 立體" and sd["sdf_data"]:
                viewer = py3Dmol.view(width=450, height=350)
                viewer.addModel(sd["sdf_data"], "sdf")
                viewer.setStyle({style: {}})
                viewer.setBackgroundColor('#f0f2f6')
                viewer.zoomTo()
                safe_html = viewer._make_html().replace("http://", "https://")
                components.html(safe_html, height=350, width=450)
            elif len(st.session_state.mol_bonds) == 0 and len(st.session_state.mol_atoms) > 0:
                st.caption("💡 查無單分子 3D 座標，系統已自動動態生成微型離子晶格模型。")
                html_content = generate_crystal_lattice_html(sd["unique_elements"], style)
                components.html(html_content, height=350, width=450)
            else:
                st.warning("⚠️ 查無官方 3D 模型，系統已降級為高解析度 2D 結構圖。")
                img_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{sd['cid']}/PNG?image_size=large"
                st.image(img_url, use_container_width=True)
            
            st.markdown("---")
            st.subheader("🧮 計算結構屬性")
            st.markdown(f"""
            * **慣用化學式:** `{sd['fixed_formula']}`
            * **分子量:** `{sd['molecular_weight']} g/mol`
            * **TPSA (極性表面積):** `{sd['tpsa']} Å²`
            * **氫鍵 (供體/受體):** `{sd['h_bond_donor_count']} / {sd['h_bond_acceptor_count']}`
            * **SMILES:** `{sd['isomeric_smiles']}`
            """)

        with col_right:
            st.subheader("⚠️ SDS 物質安全與危害標示 (GHS)")
            sds = sd['sds_data']
            signal = sds["危險信號詞"]
            if "Danger" in signal: st.error(f"**🚨 警示語: {signal}**")
            elif "Warning" in signal: st.warning(f"**⚠️ 警示語: {signal}**")
            else: st.success("**✅ 警示語: 無特殊危險標示**")
                
            if sds["危害警告"]:
                for h in sds["危害警告"]: st.caption(f"▪️ {h}")
            else: st.caption("無查獲特定危害聲明紀錄。")

            st.markdown("---")
            st.subheader("🌡️ 實驗室文獻實測數據")
            prop_md = f"""
            | 屬性類別 | 文獻實測數值 (包含單位) |
            | :--- | :--- |
            | 🧊 **密度 (Density)** | {sds["密度"]} |
            | ♨️ **沸點 (Boiling Point)** | {sds["沸點"]} |
            | ❄️ **熔點 (Melting Point)** | {sds["熔點"]} |
            | 🔥 **閃點 (Flash Point)** | {sds["閃點"]} |
            | 💧 **溶解度 (Solubility)** | {sds["溶解度"]} |
            | ☁️ **蒸氣壓 (Vapor Pressure)** | {sds["蒸氣壓"]} |
            | 👁️ **外觀與性狀** | {sds["外觀與性狀"]} |
            """
            st.markdown(prop_md)
    else:
        st.info("💡 請在左側輸入化學式或物質名稱，並按下「🔍 執行數據檢索」來啟動百科。")

# ==========================================
# 分頁 2：拉普拉斯矩陣極速動態熱傳導台 (客戶端原生動畫版)
# ==========================================
with tab2:
    st.subheader(f"🔥 {st.session_state.mol_name} - 原生硬體加速熱能擴散")

    atoms = st.session_state.mol_atoms
    bonds = st.session_state.mol_bonds
    core = st.session_state.core_node
    edge = st.session_state.edge_node
    N = len(atoms)

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 4])
    with btn_col1:
        # 點擊按鈕後，後端瞬間算完資料，傳給前端播放
        start_anim = st.button("▶️ 生成並下載動畫底片", type="primary", use_container_width=True)
    with btn_col2:
        if st.button("🔄 重置初始溫度", use_container_width=True):
            st.session_state.particle_temps = {i: env_temp for i in atoms}
            st.session_state.particle_temps[core] = init_temp
            st.rerun()

    # --- 建立空間幾何拓樸 ---
    G = nx.Graph()
    G.add_nodes_from(atoms)
    G.add_edges_from(bonds)
    
    # 防呆：無機鹽若無共價鍵，動態拉取模擬網格
    if len(bonds) == 0 and len(atoms) > 1:
        for idx in range(len(atoms) - 1):
            G.add_edge(atoms[idx], atoms[idx+1])
        bonds = list(G.edges())

    if len(atoms) > 1:
        pos_3d = nx.spring_layout(G, dim=3, seed=42)
    else:
        pos_3d = {a: [0, 0, 0] for a in atoms}
    
    edge_x, edge_y, edge_z = [], [], []
    for bond in G.edges():
        x0, y0, z0 = pos_3d[bond[0]]
        x1, y1, z1 = pos_3d[bond[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    node_x = [pos_3d[i][0] for i in atoms]
    node_y = [pos_3d[i][1] for i in atoms]
    node_z = [pos_3d[i][2] for i in atoms]

    if N > 0:
        L_matrix = nx.laplacian_matrix(G, nodelist=atoms).toarray()
    node_to_idx = {node: idx for idx, node in enumerate(atoms)}

    # ==========================================
    # 核心黑科技：瞬間預先計算所有影格 (Client-Side Animation Prep)
    # ==========================================
    if start_anim and N > 0:
        with st.spinner("⚡ 矩陣極速運算中... 正在打包 150 張電影底片給瀏覽器！"):
            m, c_heat, dt = 1.0, 1.0, 0.02
            T_array = np.array([st.session_state.particle_temps[i] for i in atoms])
            
            # 歷史數據容器
            history_frames = []
            core_hist = [T_array[node_to_idx[core]]]
            edge_hist = [T_array[node_to_idx[edge]]]
            time_hist = [0.0]
            current_t = 0.0
            
            # 🚀 後端 0.01 秒瞬間跑完 150 次拉普拉斯矩陣運算
            for frame in range(150):
                dT = -k_val * dt * (L_matrix.dot(T_array)) / (m * c_heat)
                T_array += dT
                current_t += dt
                
                # 抽取幀數減輕前端負擔 (每 2 幀存一張底片)
                if frame % 2 == 0 or frame == 149:
                    history_frames.append(T_array.copy())
                    core_hist.append(T_array[node_to_idx[core]])
                    edge_hist.append(T_array[node_to_idx[edge]])
                    time_hist.append(current_t)
            
            # 建立 Plotly 基礎圖表
            fig3d = go.Figure()
            
            # 繪製靜態化學鍵 (Trace 0)
            fig3d.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color='gray', width=3), hoverinfo='none'))
            
            # 繪製初始原子 (Trace 1)
            init_T = history_frames[0]
            init_labels = []
            for i in atoms:
                prefix = "🔥 Core" if i == core else ("❄️ Edge" if i == edge else f"Atom {i}")
                init_labels.append(f"{prefix}<br>{init_T[node_to_idx[i]]:.1f}°C")
                
            fig3d.add_trace(go.Scatter3d(
                x=node_x, y=node_y, z=node_z, mode='markers+text',
                text=init_labels, textposition="top center", textfont=dict(size=11, color='white'),
                marker=dict(size=24, color=init_T, colorscale='Turbo', cmin=-20, cmax=500,
                            colorbar=dict(title="溫度 (°C)", thickness=15), line=dict(width=2, color='white'))
            ))
            
            # 🚀 將 150 張底片封裝進 Plotly Frames (瀏覽器原生播放)
            anim_frames = []
            for idx, t_data in enumerate(history_frames):
                frame_labels = []
                for i in atoms:
                    prefix = "🔥 Core" if i == core else ("❄️ Edge" if i == edge else f"Atom {i}")
                    frame_labels.append(f"{prefix}<br>{t_data[node_to_idx[i]]:.1f}°C")
                    
                anim_frames.append(go.Frame(
                    data=[
                        go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color='gray', width=3)),
                        go.Scatter3d(x=node_x, y=node_y, z=node_z, mode='markers+text', marker=dict(color=t_data), text=frame_labels)
                    ],
                    name=f"frame_{idx}"
                ))
            fig3d.frames = anim_frames
            
            # 加上內建播放器 UI
            fig3d.update_layout(
                title="🎬 電影級動畫已準備就緒 (請點擊左下角 ▶️ 播放)",
                scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
                margin=dict(l=0, r=0, b=0, t=40), height=480, template="plotly_dark", showlegend=False,
                updatemenus=[dict(
                    type="buttons", showactive=False, y=-0.1, x=0.1, xanchor="right", yanchor="top",
                    buttons=[
                        dict(label="▶️ 播放動畫", method="animate", args=[None, dict(frame=dict(duration=anim_speed, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                        dict(label="⏸️ 暫停", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])
                    ]
                )]
            )

            # 繪製靜態總結折線圖
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(x=time_hist, y=core_hist, mode='lines', name=f'中心點火源 (Atom {core})', line=dict(color='red', width=3)))
            fig_line.add_trace(go.Scatter(x=time_hist, y=edge_hist, mode='lines', name=f'最外圍原子 (Atom {edge})', line=dict(color='blue', width=3)))
            fig_line.update_layout(
                title="📈 系統最終溫度趨勢全覽 (°C)", xaxis_title="時間 (秒)", yaxis_title="溫度 (°C)",
                margin=dict(l=0, r=0, b=0, t=40), height=480, template="plotly_dark"
            )

            # 一次性拋出渲染，不佔用後端資源
            col_visual, col_chart = st.columns([1.4, 1])
            with col_visual: st.plotly_chart(fig3d, use_container_width=True)
            with col_chart: st.plotly_chart(fig_line, use_container_width=True)
            
            # 更新狀態機至最終結果
            for i in atoms: st.session_state.particle_temps[i] = T_array[node_to_idx[i]]

            st.success("✅ 運算與封裝完成！請點擊 3D 圖表左下角的「▶️ 播放動畫」按鈕觀賞絲滑體驗。")

    else:
        # --- 初始/靜態狀態渲染 ---
        col_visual, col_chart = st.columns([1.4, 1])
        
        node_colors = [st.session_state.particle_temps[i] for i in atoms]
        node_labels = []
        for i in atoms:
            prefix = "🔥 Core" if i == core else ("❄️ Edge" if i == edge else f"Atom {i}")
            node_labels.append(f"{prefix}<br>{st.session_state.particle_temps[i]:.1f}°C")
            
        fig3d = go.Figure()
        fig3d.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color='gray', width=3), hoverinfo='none'))
        fig3d.add_trace(go.Scatter3d(
            x=node_x, y=node_y, z=node_z, mode='markers+text',
            text=node_labels, textposition="top center", textfont=dict(size=11, color='white'),
            marker=dict(size=24, color=node_colors, colorscale='Turbo', cmin=-20, cmax=500,
                        colorbar=dict(title="溫度 (°C)", thickness=15), line=dict(width=2, color='white'))
        ))
        fig3d.update_layout(
            title=f"🛑 目前狀態為靜止 (請點擊左上方按鈕生成動畫)",
            scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
            margin=dict(l=0, r=0, b=0, t=40), height=480, template="plotly_dark", showlegend=False
        )
        with col_visual: st.plotly_chart(fig3d, use_container_width=True)

        fig_line = go.Figure()
        fig_line.update_layout(
            title="📈 核心與外圍原子溫度趨勢 (°C)", xaxis_title="時間 (秒)", yaxis_title="溫度 (°C)",
            margin=dict(l=0, r=0, b=0, t=40), height=480, template="plotly_dark"
        )
        with col_chart: st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("### 🔢 每個粒子的即時溫度數據面板 (°C)")
    df_realtime = pd.DataFrame({
        "粒子編號": [f"Atom {i}" for i in atoms],
        "即時溫度數據 (°C)": [f"{st.session_state.particle_temps[i]:.2f} °C" for i in atoms],
        "拓樸定位": ["🔥 中心點火源" if i == core else "❄️ 外部邊緣節點" if i == edge else "中圈熱傳導節點" for i in atoms]
    })
    st.dataframe(df_realtime.set_index("粒子編號"), use_container_width=True)
