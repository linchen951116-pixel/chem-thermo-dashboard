import streamlit as st
import pubchempy as pcp
import py3Dmol
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.linalg import expm
from deep_translator import GoogleTranslator
import re
import requests
import streamlit.components.v1 as components

# 網頁基礎設定 (寬版佈局)
st.set_page_config(page_title="中文化學物質分析與動態熱力學系統", layout="wide")

st.title("🧪 物質深度分析 & 3D 動態熱力學系統")
st.markdown("搭載 **矩陣指數絕對精確解** 與 **中央動態聯動播放引擎**，實現 3D 模型、2D 折線圖與數據表格之毫秒級完美同步。")

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
        headers = {'User-Agent': 'Mozilla/5.0'}
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
            if not formula.startswith("C"): formula = metal + formula.replace(metal, "")
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
    props = {"外觀與性狀": "無相關文獻數據", "密度": "無相關文獻數據", "熔點": "無相關文獻數據", "沸點": "無相關文獻數據", "閃點": "無相關文獻數據", "溶解度": "無相關文獻數據", "蒸氣壓": "無相關文獻數據", "危險信號詞": "無標示 / 安全", "危害警告": []}
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
                if (x + y + z) % 4 == 0: current_color, r = c1, 0.9 if style == "sphere" else 0.5
                else: current_color, r = c2, 1.2 if style == "sphere" else 0.6
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
    user_input = st.text_input("輸入化學式、中文試劑或藥品名稱", "Benzene").strip()
    style = st.selectbox("3D 顯示風格", ["stick", "sphere", "line", "cross"])
    search_button = pcp_search = st.button("🔍 執行數據檢索", type="primary")
    
    st.markdown("---")
    st.subheader("🔥 2. 熱傳導動態模擬參數")
    env_temp = st.slider("環境溫度設定 (°C)", min_value=-20.0, max_value=60.0, value=25.0, step=0.5)
    init_temp = st.slider("中心粒子點火溫度 (°C)", min_value=50.0, max_value=500.0, value=500.0, step=10.0) # 預設直接拉滿500°C
    k_val = st.slider("熱傳導係數 (k)", min_value=0.01, max_value=0.50, value=0.15, step=0.01)
    sim_duration = st.slider("模擬總時長 (秒)", min_value=3.0, max_value=30.0, value=10.0, step=1.0)
    sim_speed = st.slider("幀率刷新延遲 (秒)", min_value=0.01, max_value=0.30, value=0.05, step=0.01)

# ==========================================
# 核心狀態機管理 (Session State)
# ==========================================
if 'mol_atoms' not in st.session_state:
    st.session_state.mol_atoms = list(range(6))
    st.session_state.mol_bonds = [(1,2), (2,3), (3,4), (4,5), (5,6), (6,1)]
    st.session_state.core_node = 1
    st.session_state.edge_node = 4
    st.session_state.mol_name = "苯 (預設結構)"

if 'history_frames' not in st.session_state:
    st.session_state.history_frames = None
    st.session_state.time_steps = None

# 滑桿智慧監聽同步
if 'last_env' not in st.session_state: st.session_state.last_env = env_temp
if 'last_init' not in st.session_state: st.session_state.last_init = init_temp

if st.session_state.last_env != env_temp or st.session_state.last_init != init_temp:
    st.session_state.history_frames = None  # 參數變更時清除舊底片，強制重新生成
    st.session_state.last_env = env_temp
    st.session_state.last_init = init_temp

# 智慧防錯數據檢索
if pcp_search and user_input:
    with st.spinner("🧠 系統正在調閱學術詞典並重建分子拓樸..."):
        try:
            english_name = user_input
            if contains_chinese(user_input) or user_input in LOCAL_CHEM_DICT:
                if user_input in LOCAL_CHEM_DICT: english_name = LOCAL_CHEM_DICT[user_input]
                else:
                    wiki_name = translate_via_wikipedia(user_input)
                    if wiki_name: english_name = wiki_name
                    else:
                        translated = GoogleTranslator(source='auto', target='en').translate(user_input)
                        if contains_chinese(translated):
                            st.warning(f"⚠️ 無法自動辨識「{user_input}」")
                            st.stop()
                        else: english_name = translated

            compounds = pcp.get_compounds(english_name, 'name')
            if not compounds:
                st.warning(f"⚠️ 無法精確配對「{english_name}」，請檢查拼字。")
            else:
                c = compounds[0]
                atoms = [atom.aid for atom in c.atoms]
                bonds = [(bond.aid1, bond.aid2) for bond in c.bonds]
                
                degree = {}
                for a, b in bonds:
                    degree[a] = degree.get(a, 0) + 1
                    degree[b] = degree.get(b, 0) + 1
                c_node = max(degree, key=degree.get) if degree else (atoms[0] if atoms else 1)
                e_node = min(degree, key=degree.get) if degree else (atoms[-1] if atoms else 1)
                
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
                st.session_state.history_frames = None # 清空舊物質的動畫
                
        except Exception as e:
            st.warning("⚠️ 檢索異常，請確認拼寫是否正確。")

# 雙分頁介面
tab1, tab2 = st.tabs(["🧬 SDS 物質安全與化學百科", "🔥 雙聯動極速動畫台"])

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
                components.html(viewer._make_html().replace("http://", "https://"), height=350, width=450)
            elif len(st.session_state.mol_bonds) == 0 and len(st.session_state.mol_atoms) > 0:
                html_content = generate_crystal_lattice_html(sd["unique_elements"], style)
                components.html(html_content, height=350, width=450)
            else:
                st.image(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{sd['cid']}/PNG?image_size=large", use_container_width=True)
            
            st.markdown("---")
            st.subheader("🧮 計算結構屬性")
            st.markdown(f"* **慣用化學式:** `{sd['fixed_formula']}`\n* **分子量:** `{sd['molecular_weight']} g/mol`\n* **TPSA:** `{sd['tpsa']} Å²`\n* **氫鍵 (供/受):** `{sd['h_bond_donor_count']} / {sd['h_bond_acceptor_count']}`\n* **SMILES:** `{sd['isomeric_smiles']}`")
        with col_right:
            st.subheader("⚠️ SDS 物質安全與危害標示 (GHS)")
            sds = sd['sds_data']
            if "Danger" in sds["危險信號詞"] or "危險" in sds["危險信號詞"]: st.error(f"**🚨 警示語: {sds['危險信號詞']}**")
            elif "Warning" in sds["危險信號詞"] or "警告" in sds["危險信號詞"]: st.warning(f"**⚠️ 警示語: {sds['危險信號詞']}**")
            else: st.success("**✅ 警示語: 無特殊危險標示**")
            if sds["危害警告"]:
                for h in sds["危害警告"]: st.caption(f"▪️ {h}")
            st.markdown("---")
            st.subheader("🌡️ 實驗室文獻實測數據")
            st.markdown(f"| 屬性類別 | 文獻實測數值 (包含單位) |\n| :--- | :--- |\n| 🧊 密度 | {sds['密度']} |\n| ♨️ 沸點 | {sds['沸點']} |\n| ❄️ 熔點 | {sds['熔點']} |\n| 🔥 閃點 | {sds['閃點']} |\n| 💧 溶解度 | {sds['溶解度']} |\n| ☁️ 蒸氣壓 | {sds['蒸氣壓']} |\n| 👁️ 外觀與性狀 | {sds['外觀與性狀']} |")
    else:
        st.info("💡 請在左側輸入化學物名稱，並按下「🔍 執行數據檢索」來啟動百科。")

# ==========================================
# 分頁 2：全新中央控制聯動模擬桌
# ==========================================
with tab2:
    st.subheader(f"🔥 {st.session_state.mol_name} - 動態聯動熱能擴散控制台")
    
    atoms = st.session_state.mol_atoms
    bonds = st.session_state.mol_bonds
    core = st.session_state.core_node
    edge = st.session_state.edge_node
    N = len(atoms)
    
    # 建立幾何圖論拓樸座標
    G = nx.Graph()
    G.add_nodes_from(atoms)
    G.add_edges_from(bonds)
    if len(bonds) == 0 and len(atoms) > 1:
        for idx in range(len(atoms) - 1): G.add_edge(atoms[idx], atoms[idx+1])
        bonds = list(G.edges())
    pos_3d = nx.spring_layout(G, dim=3, seed=42) if len(atoms) > 1 else {a: [0,0,0] for a in atoms}
    
    edge_x, edge_y, edge_z = [], [], []
    for b in G.edges():
        x0, y0, z0 = pos_3d[b[0]]
        x1, y1, z1 = pos_3d[b[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
        
    node_x = [pos_3d[i][0] for i in atoms]
    node_y = [pos_3d[i][1] for i in atoms]
    node_z = [pos_3d[i][2] for i in atoms]
    node_to_idx = {node: idx for idx, node in enumerate(atoms)}

    # 操作控制按鈕區
    ctrl_col1, ctrl_col2, _ = st.columns([1.2, 1.2, 3.6])
    with ctrl_col1:
        build_package = st.button("⚙️ 生成高畫質動畫底片", type="primary", use_container_width=True)
    
    # 點擊「生成底片」：瞬間預算 100 幀矩陣指數，並維持在第 0 秒靜止狀態
    if build_package and N > 0:
        with st.spinner("⚡ 矩陣微積分運算中... 正在建立絕對精確解底片"):
            m, c_heat = 1.0, 1.0
            L_matrix = nx.laplacian_matrix(G, nodelist=atoms).toarray()
            T_initial = np.array([env_temp if i != core else init_temp for i in atoms])
            
            # 建立 100 個時間影格
            st.session_state.time_steps = np.linspace(0, sim_duration, num=100)
            frames_list = []
            
            for t in st.session_state.time_steps:
                transition_matrix = expm(-k_val * t / (m * c_heat) * L_matrix)
                frames_list.append(transition_matrix.dot(T_initial))
                
            st.session_state.history_frames = frames_list
            st.success(f"✅ 影格底片封裝成功！初始點火源已精確設定為 **{init_temp:.1f}°C**。請按下方播放鈕開始展示。")

    # 建立動態更新專用的中央空容器
    dashboard_placeholder = st.empty()
    table_placeholder = st.empty()

    # 輔助渲染函數：吃單一影格資料，畫出此時此刻的聯動圖表與表格
    def render_single_frame(step_index):
        t_data = st.session_state.history_frames[step_index]
        curr_time = st.session_state.time_steps[step_index]
        
        # 準備 3D 圖數據
        labels = [f"🔥 中心源: {t_data[node_to_idx[i]]:.1f}°C" if i == core else (f"❄️ 外圍: {t_data[node_to_idx[i]]:.1f}°C" if i == edge else f"原子 {i}: {t_data[node_to_idx[i]]:.1f}°C") for i in atoms]
        
        fig = make_subplots(
            rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'xy'}]], column_widths=[0.55, 0.45],
            subplot_titles=(f"⚛️ {st.session_state.mol_name} 空間傳導", f"📈 溫度隨時間演化歷程 (累積時間: {curr_time:.2f}秒)")
        )
        # 3D 化學鍵與粒子
        fig.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color='gray', width=3), hoverinfo='none'), row=1, col=1)
        fig.add_trace(go.Scatter3d(x=node_x, y=node_y, z=node_z, mode='markers+text', text=labels, textposition="top center", textfont=dict(size=11, color='white'), marker=dict(size=22, color=t_data, colorscale='Turbo', cmin=env_temp-5, cmax=init_temp+5, colorbar=dict(title="溫度 (°C)", thickness=12, x=0.46))), row=1, col=1)
        
        # 2D 折線圖歷程
        past_times = st.session_state.time_steps[:step_index+1]
        core_y = [f[node_to_idx[core]] for f in st.session_state.history_frames[:step_index+1]]
        edge_y_vals = [f[node_to_idx[edge]] for f in st.session_state.history_frames[:step_index+1]]
        
        fig.add_trace(go.Scatter(x=past_times, y=core_y, mode='lines', name='中心點火源', line=dict(color='red', width=3)), row=1, col=2)
        fig.add_trace(go.Scatter(x=past_times, y=edge_y_vals, mode='lines', name='最外圍原子', line=dict(color='blue', width=3)), row=1, col=2)
        
        fig.update_layout(scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False), template="plotly_dark", margin=dict(l=0, r=0, b=0, t=40), height=550, uirevision='constant')
        fig.update_xaxes(range=[0, sim_duration], title="時間 (秒)", row=1, col=2)
        fig.update_yaxes(range=[env_temp - 10, init_temp + 20], title="溫度 (°C)", row=1, col=2)
        
        # 建立同步數據表格
        df = pd.DataFrame({
            "粒子編號": [f"Atom {i}" for i in atoms],
            "即時溫度 (°C)": [f"{t_data[node_to_idx[i]]:.2f} °C" for i in atoms],
            "拓樸定位": ["🔥 中心點火源" if i == core else "❄️ 外部邊緣節點" if i == edge else "中圈傳導節點" for i in atoms]
        }).set_index("粒子編號")
        
        # 推播送到主要容器中
        dashboard_placeholder.plotly_chart(fig, use_container_width=True, key="stable_sync_dashboard")
        table_placeholder.dataframe(df, use_container_width=True)

    # 渲染邏輯狀態機
    if st.session_state.history_frames is not None:
        # 如果已經生成資料，提供單獨的播放按鈕，且「預設停在第0幀」絕不自動播放
        with ctrl_col2:
            trigger_play = st.button("▶️ 開始播放聯動動畫", use_container_width=True)
            
        if trigger_play:
            # 執行聯動動畫大巡航迴圈
            for step in range(len(st.session_state.time_steps)):
                render_single_frame(step)
                time.sleep(sim_speed)
        else:
            # 靜止待命狀態下，安靜地定格在第一幀 (0.00 秒)，完美呈現滑桿設定的 500°C
            render_single_frame(0)
    else:
        st.info("💡 請點擊上方『⚙️ 生成高畫質動畫底片』。系統將會瞬間預算完畢並在第 0 秒靜止待命，等候您手動點擊播放。")
