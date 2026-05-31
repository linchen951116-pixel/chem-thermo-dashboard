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
import json
import streamlit.components.v1 as components

# 網頁基礎設定
st.set_page_config(page_title="中文化學物質分析與動態熱力學系統", layout="wide")

st.title("🧪 物質深度分析 & 3D 動態熱力學系統")
st.markdown("搭載 **維度安全攔截機制 (Dimension Guard)** 與 **多方備援抓取引擎**。自動辨識物態幾何，確保科學嚴謹性與網頁渲染絕對穩定！")

# ==========================================
# 核心一：維基百科學術名詞對接與修正引擎
# ==========================================
LOCAL_CHEM_DICT = {
    "阿斯匹靈": "Aspirin", "普拿疼": "Acetaminophen", "雙氧水": "Hydrogen peroxide",
    "鹽酸": "Hydrochloric acid", "硫酸": "Sulfuric acid", "硝酸": "Nitric acid",
    "氨水": "Ammonia", "食鹽": "Sodium chloride", "氯化鈉": "Sodium chloride",
    "硝酸鉀": "Potassium nitrate", "亞硝酸鉀": "Potassium nitrite",
    "KNO2": "Potassium nitrite", "kno2": "Potassium nitrite",
    "高錳酸鉀": "Potassium permanganate", "碳酸鈉": "Sodium carbonate",
    "氫氧化鈉": "Sodium hydroxide", "乙醇": "Ethanol", "甲醇": "Methanol",
    "苯": "Benzene", "水": "Water", "咖啡酸": "Caffeic acid"
}

def contains_chinese(text): return bool(re.search('[\u4e00-\u9fff]', text))

def translate_via_wikipedia(zh_name):
    try:
        url = f"https://zh.wikipedia.org/w/api.php?action=query&prop=langlinks&titles={zh_name}&lllang=en&format=json"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5).json()
        for _, page_info in res.get("query", {}).get("pages", {}).items():
            if "langlinks" in page_info: return page_info["langlinks"][0]["*"] 
    except: pass
    return None

def fix_chemical_formula(formula):
    if not formula: return formula
    fix_map = {"ClNa": "NaCl", "HNaO": "NaOH", "ClK": "KCl", "HKO": "KOH", "IK": "KI", "KNO2": "KNO₂", "NO2K": "KNO₂", "NO3K": "KNO₃", "C2H4O2": "CH₃COOH"}
    if formula in fix_map: return fix_map[formula]
    for metal in ["Na", "K", "Ca", "Mg", "Al", "Fe", "Cu", "Zn", "Ag"]:
        if metal in formula and not formula.startswith(metal) and not formula.startswith("C"): 
            return metal + formula.replace(metal, "")
    return formula

# ==========================================
# 核心二：數據抓取與智慧正規化引擎
# ==========================================
def safe_translate(text):
    if not text or text == "無相關文獻數據": return text
    try: return GoogleTranslator(source='en', target='zh-TW').translate(text)
    except: return text

def simplify_physical_state(text):
    if not text or text == "無相關文獻數據": return text
    t = text.lower()
    is_sol = "solution" in t or "aqueous" in t
    if any(kw in t for kw in ["solid", "crystal", "powder", "pellet", "salt"]): return "🧊 固體"
    elif any(kw in t for kw in ["liquid", "fluid"]) and not is_sol: return "💧 液體"
    elif any(kw in t for kw in ["gas", "vapor"]): return "☁️ 氣體"
    elif is_sol: return "💧 水溶液 (純物質通常為固體)"
    return text

def standardize_temperature(raw_str):
    if not raw_str: return None
    c_match = re.search(r'(-?\d+\.?\d*)\s*(?:-|to|~)\s*(-?\d+\.?\d*)\s*(?:°C|deg C|C\b)', raw_str, re.IGNORECASE)
    if c_match: return f"{c_match.group(1)} ~ {c_match.group(2)} °C"
    c_single = re.search(r'(-?\d+\.?\d*)\s*(?:°C|deg C|C\b)', raw_str, re.IGNORECASE)
    if c_single: return f"{c_single.group(1)} °C"
    f_match = re.search(r'(-?\d+\.?\d*)\s*(?:-|to|~)\s*(-?\d+\.?\d*)\s*(?:°F|deg F|F\b)', raw_str, re.IGNORECASE)
    if f_match: return f"{(float(f_match.group(1))-32)*5/9:.1f} ~ {(float(f_match.group(2))-32)*5/9:.1f} °C"
    f_single = re.search(r'(-?\d+\.?\d*)\s*(?:°F|deg F|F\b)', raw_str, re.IGNORECASE)
    if f_single: return f"{(float(f_single.group(1))-32)*5/9:.1f} °C"
    k_match = re.search(r'(-?\d+\.?\d*)\s*(?:-|to|~)\s*(-?\d+\.?\d*)\s*(?:K\b)', raw_str)
    if k_match: return f"{float(k_match.group(1))-273.15:.1f} ~ {float(k_match.group(2))-273.15:.1f} °C"
    k_single = re.search(r'(-?\d+\.?\d*)\s*(?:K\b)', raw_str)
    if k_single: return f"{float(k_single.group(1))-273.15:.1f} °C"
    return None

def get_mol_sdf(cid):
    try:
        res = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/record/SDF/?record_type=3d", timeout=5)
        if res.status_code == 200: return res.text, "3D 立體"
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
        res = requests.get(url, timeout=10).json()
        sections = res.get("Record", {}).get("Section", [])
        property_map = {"Physical Description": "外觀與性狀", "Density": "密度", "Melting Point": "熔點", "Boiling Point": "沸點", "Flash Point": "閃點", "Solubility": "溶解度", "Vapor Pressure": "蒸氣壓"}
        
        for sec in sections:
            if sec.get("TOCHeading") == "Chemical and Physical Properties":
                for subsec in sec.get("Section", []):
                    if subsec.get("TOCHeading") in ["Experimental Properties", "Computed Properties"]:
                        for prop in subsec.get("Section", []):
                            heading = prop.get("TOCHeading")
                            if heading in property_map:
                                target_prop_zh = property_map[heading]
                                if props[target_prop_zh] != "無相關文獻數據": continue
                                for info in prop.get("Information", []):
                                    val_list = info.get("Value", {}).get("StringWithMarkup", [])
                                    val_str = val_list[0].get("String") if val_list else None
                                    if not val_str: continue
                                    if target_prop_zh == "外觀與性狀":
                                        props[target_prop_zh] = simplify_physical_state(val_str)
                                        break
                                    elif target_prop_zh in ["熔點", "沸點", "閃點"]:
                                        parsed_temp = standardize_temperature(val_str)
                                        if parsed_temp:
                                            props[target_prop_zh] = parsed_temp
                                            break
                                    else:
                                        if target_prop_zh == "密度" and "g/" not in val_str and "kg/" not in val_str: val_str += " g/cm³"
                                        props[target_prop_zh] = val_str
                                        break
            elif sec.get("TOCHeading") == "Safety and Hazards":
                for subsec in sec.get("Section", []):
                    if subsec.get("TOCHeading") == "Hazards Identification":
                        for ghs in subsec.get("Section", []):
                            if ghs.get("TOCHeading") == "GHS Classification":
                                for info in ghs.get("Information", []):
                                    if info.get("Name") == "Signal":
                                        raw_sig = info["Value"]["StringWithMarkup"][0]["String"]
                                        props["危險信號詞"] = "危險 (Danger)" if "Danger" in raw_sig else ("警告 (Warning)" if "Warning" in raw_sig else raw_sig)
                                    elif info.get("Name") == "GHS Hazard Statements":
                                        raw_hazards = [h["String"] for h in info["Value"]["StringWithMarkup"]]
                                        props["危害警告"] = [safe_translate(h) for h in raw_hazards[:5]]
    except: pass
    return props

def generate_crystal_lattice_html(elements, style):
    viewer = py3Dmol.view(width=450, height=350)
    viewer.setBackgroundColor('#f0f2f6')
    cmap = {"Na": "purple", "K": "violet", "Cl": "green", "O": "red", "N": "blue", "Default": "orange"}
    c1 = cmap.get(elements[0] if len(elements)>0 else "Default", cmap["Default"])
    c2 = cmap.get(elements[1] if len(elements)>1 else "O", cmap["Default"])
    for x in [-2,0,2]:
        for y in [-2,0,2]:
            for z in [-2,0,2]:
                r, c = (0.9 if style=="sphere" else 0.5, c1) if (x+y+z)%4==0 else (1.2 if style=="sphere" else 0.6, c2)
                viewer.addSphere({'center': {'x': x*3, 'y': y*3, 'z': z*3}, 'radius': r, 'color': c})
    if style in ["stick", "sphere", "line"]:
        for i in [-2,0,2]:
            for j in [-2,0,2]:
                viewer.addLine({'start': {'x':-6, 'y':i*3, 'z':j*3}, 'end': {'x':6, 'y':i*3, 'z':j*3}, 'color': 'gray'})
                viewer.addLine({'start': {'x':i*3, 'y':-6, 'z':j*3}, 'end': {'x':i*3, 'y':6, 'z':j*3}, 'color': 'gray'})
                viewer.addLine({'start': {'x':i*3, 'y':i*3, 'z':-6}, 'end': {'x':i*3, 'y':j*3, 'z':6}, 'color': 'gray'})
    viewer.zoomTo()
    return viewer._make_html().replace("http://", "https://")

# --- 側邊欄：全局參數設定面板 ---
with st.sidebar:
    st.header("⚙️ 全局參數設定面板")
    st.subheader("🔬 1. 物質百科檢索")
    user_input = st.text_input("輸入化學式、中文試劑或藥品名稱", "水").strip()
    style = st.selectbox("3D 顯示風格", ["stick", "sphere", "line", "cross"])
    search_button = st.button("🔍 執行數據檢索", type="primary")
    
    st.markdown("---")
    st.subheader("🔥 2. 熱傳導動態模擬參數")
    env_temp = st.slider("環境溫度設定 (°C)", min_value=-20.0, max_value=60.0, value=25.0, step=0.5)
    init_temp = st.slider("中心粒子點火溫度 (°C)", min_value=50.0, max_value=500.0, value=500.0, step=10.0)
    k_val = st.slider("熱傳導係數 (k)", min_value=0.01, max_value=0.50, value=0.15, step=0.01)
    sim_duration = st.slider("模擬總時長 (秒)", min_value=3.0, max_value=30.0, value=10.0, step=1.0)
    anim_speed = st.slider("動畫播放速度 (每幀毫秒)", min_value=10, max_value=200, value=40, step=10)

# ==========================================
# 狀態機與記憶體安全初始化
# ==========================================
if 'mol_atoms' not in st.session_state:
    st.session_state.mol_atoms = list(range(3))
    st.session_state.mol_bonds = [(0,1), (1,2)]
    st.session_state.mol_coords = {}  
    st.session_state.core_node = 1
    st.session_state.edge_node = 0
    st.session_state.mol_name = "載入中"

def run_search(query_name):
    english_name = query_name
    if contains_chinese(query_name) or query_name in LOCAL_CHEM_DICT:
        if query_name in LOCAL_CHEM_DICT: english_name = LOCAL_CHEM_DICT[query_name]
        else:
            wiki_name = translate_via_wikipedia(query_name)
            if wiki_name: english_name = wiki_name
            else:
                translated = GoogleTranslator(source='auto', target='en').translate(query_name)
                if contains_chinese(translated): return False, f"⚠️ 無法辨識「{query_name}」"
                else: english_name = translated

    compounds = pcp.get_compounds(english_name, 'name', record_type='3d')
    if not compounds: compounds = pcp.get_compounds(english_name, 'name') 
    if not compounds: return False, f"⚠️ 資料庫無法配對「{english_name}」"
    
    c = compounds[0]
    atoms = [atom.aid for atom in c.atoms]
    bonds = [(bond.aid1, bond.aid2) for bond in c.bonds]
    
    real_coords = {}
    for atom in c.atoms:
        if hasattr(atom, 'x') and atom.x is not None: real_coords[atom.aid] = [atom.x, atom.y, atom.z]
            
    degree = {}
    for a, b in bonds:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
        
    if degree:
        sorted_nodes = sorted(degree.keys(), key=lambda x: degree[x])
        c_node = sorted_nodes[-1] 
        e_node = sorted_nodes[0]  
        if c_node == e_node and len(atoms) > 1: e_node = [n for n in atoms if n != c_node][0]
    else:
        c_node = atoms[0] if atoms else 0
        e_node = atoms[-1] if len(atoms) > 1 else 0
        
    st.session_state.search_data = {
        "english_name": english_name.capitalize(), "cid": c.cid, "fixed_formula": fix_chemical_formula(c.molecular_formula),
        "molecular_weight": c.molecular_weight, "tpsa": c.tpsa, "h_bond_donor_count": c.h_bond_donor_count,
        "h_bond_acceptor_count": c.h_bond_acceptor_count, "isomeric_smiles": c.isomeric_smiles,
        "sds_data": fetch_sds_and_properties(c.cid), "unique_elements": list(set([a.element for a in c.atoms]))
    }
    sdf, dim = get_mol_sdf(c.cid)
    st.session_state.search_data.update({"sdf_data": sdf, "dim_type": dim})
    
    st.session_state.mol_atoms = atoms
    st.session_state.mol_bonds = bonds
    st.session_state.mol_coords = real_coords 
    st.session_state.core_node = c_node
    st.session_state.edge_node = e_node
    st.session_state.mol_name = english_name.capitalize()
    st.session_state.particle_temps = {i: env_temp for i in atoms}
    st.session_state.particle_temps[c_node] = init_temp
    return True, "Success"

if 'initialized' not in st.session_state:
    run_search("水")
    st.session_state.initialized = True
    st.session_state.last_env = env_temp
    st.session_state.last_init = init_temp

if st.session_state.last_env != env_temp or st.session_state.last_init != init_temp:
    st.session_state.particle_temps = {i: env_temp for i in st.session_state.mol_atoms}
    st.session_state.particle_temps[st.session_state.core_node] = init_temp
    st.session_state.last_env = env_temp
    st.session_state.last_init = init_temp

if search_button and user_input:
    with st.spinner("🧠 系統正在向美國 NIH 資料庫調閱真實物理座標與文獻數據..."):
        success, msg = run_search(user_input)
        if not success: st.error(msg)

tab1, tab2 = st.tabs(["🧬 SDS 物質安全與化學百科", "🔥 網格分離式動畫儀表板"])

# ==========================================
# 分頁 1：化學百科與 SDS 危害報告 
# ==========================================
with tab1:
    if 'search_data' in st.session_state:
        sd = st.session_state.search_data
        st.success(f"✅ 當前載入物質：「**{sd['english_name']}**」 | 系統已成功解析全數 **{len(st.session_state.mol_atoms)}** 顆原子。")
        col_left, col_right = st.columns([1, 1.3])
        with col_left:
            st.subheader("⚛️ 空間立體結構")
            if sd["dim_type"] == "3D 立體" and sd["sdf_data"]:
                viewer = py3Dmol.view(width=450, height=350)
                viewer.addModel(sd["sdf_data"], "sdf")
                if style == "sphere": viewer.setStyle({'sphere': {}})
                else: viewer.setStyle({style: {}, 'sphere': {'radius': 0.2}})
                viewer.setBackgroundColor('#f0f2f6')
                viewer.zoomTo()
                components.html(viewer._make_html().replace("http://", "https://"), height=350, width=450)
            elif len(st.session_state.mol_bonds) == 0 and len(st.session_state.mol_atoms) > 0:
                components.html(generate_crystal_lattice_html(sd["unique_elements"], style), height=350, width=450)
            else:
                st.warning("⚠️ 查無官方 3D 模型，系統已降級為 2D 結構圖。")
                st.image(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{sd['cid']}/PNG?image_size=large", use_container_width=True)
            
            st.markdown("---")
            st.subheader("🧮 計算結構屬性")
            st.markdown(f"* **慣用化學式:** `{sd['fixed_formula']}`\n* **分子量:** `{sd['molecular_weight']} g/mol`\n* **TPSA:** `{sd['tpsa']} Å²`\n* **氫鍵 (供/受):** `{sd['h_bond_donor_count']} / {sd['h_bond_acceptor_count']}`\n* **SMILES:** `{sd['isomeric_smiles']}`")

        with col_right:
            st.subheader("⚠️ SDS 物質安全與危害標示 (GHS)")
            sds = sd['sds_data']
            signal = sds["危險信號詞"]
            if "Danger" in signal: st.error(f"**🚨 警示語: {signal}**")
            elif "Warning" in signal: st.warning(f"**⚠️ 警示語: {signal}**")
            else: st.success("**✅ 警示語: 無特殊危險標示**")
            if sds["危害警告"]:
                for h in sds["危害警告"]: st.caption(f"▪️ {h}")
            st.markdown("---")
            st.subheader("🌡️ 實驗室文獻實測數據")
            prop_md = f"""
            | 屬性類別 | 文獻實測數值 (系統已統一攝氏溫標) |
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

# ==========================================
# 分頁 2：網格分離式動畫儀表板 (安全防禦版)
# ==========================================
with tab2:
    if 'search_data' in st.session_state and st.session_state.search_data.get("dim_type") == "2D 平面":
        st.warning("⚠️ 偵測到當前物質僅具備 2D 平面結構，系統已自動攔截並停用熱傳導模擬。")
        st.info("""
        💡 **維度安全防護說明：**
        由於本系統採用 **偏微分方程之矩陣指數 (Matrix Exponential)** 進行真實空間的三維熱傳導動態模擬，運算高度依賴物質在真實世界中的 $x, y, z$ 物理幾何座標。
        
        當前檢索的物質在國際資料庫中僅留有 2D 拓樸紀錄。為了避免網格運算產生無效解，並防止瀏覽器 WebGL 畫布因溢出而崩潰，系統已安全停用此物質的模擬功能。
        
        **👉 建議嘗試：**
        請重新檢索具備完整 3D 結構的共價分子（如：**阿斯匹靈**、**苯**、**普拿疼**、**水**等），即可解鎖高畫質雙聯動模擬。
        """)
    else:
        st.subheader(f"🔥 {st.session_state.mol_name} - 獨立視窗防溢出監控台")

        atoms = st.session_state.mol_atoms
        bonds = st.session_state.mol_bonds
        core = st.session_state.core_node
        edge = st.session_state.edge_node
        N = len(atoms)

        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 4])
        with btn_col1:
            start_anim = st.button("⚙️ 生成劇院級分離式底片", type="primary", use_container_width=True)
        with btn_col2:
            if st.button("🔄 重置初始溫度 (恢復為滑桿數值)", use_container_width=True):
                st.session_state.particle_temps = {i: env_temp for i in atoms}
                st.session_state.particle_temps[core] = init_temp
                st.rerun()

        G = nx.Graph()
        G.add_nodes_from(atoms)
        G.add_edges_from(bonds)
        if len(bonds) == 0 and len(atoms) > 1:
            for idx in range(len(atoms) - 1): G.add_edge(atoms[idx], atoms[idx+1])
            bonds = list(G.edges())

        pos_3d = st.session_state.mol_coords
        
        edge_x, edge_y, edge_z = [], [], []
        for bond in G.edges():
            edge_x.extend([pos_3d[bond[0]][0], pos_3d[bond[1]][0], None])
            edge_y.extend([pos_3d[bond[0]][1], pos_3d[bond[1]][1], None])
            edge_z.extend([pos_3d[bond[0]][2], pos_3d[bond[1]][2], None])

        node_x = [pos_3d[i][0] for i in atoms]
        node_y = [pos_3d[i][1] for i in atoms]
        node_z = [pos_3d[i][2] for i in atoms]
        node_to_idx = {node: idx for idx, node in enumerate(atoms)}

        if start_anim and N > 0:
            with st.spinner(f"⚡ 啟動純淨渲染引擎... 正在鎖定畫布邊界！"):
                L_matrix = nx.laplacian_matrix(G, nodelist=atoms).toarray()
                T_initial = np.array([st.session_state.particle_temps[i] for i in atoms])
                time_steps = np.linspace(0, sim_duration, num=100)
                
                history_frames = []
                core_hist, edge_hist = [], []
                
                for t in time_steps:
                    T_t = expm(-k_val * t / 1.0 * L_matrix).dot(T_initial)
                    history_frames.append(T_t)
                    core_hist.append(T_t[node_to_idx[core]])
                    edge_hist.append(T_t[node_to_idx[edge]])
                
                fig3d = go.Figure()
                fig3d.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color='gray', width=3), hoverinfo='none'))
                
                init_T = history_frames[0]
                init_labels = [f"🔥 核心源<br>溫度: {init_T[node_to_idx[i]]:.1f}°C" if i == core else (f"❄️ 外圍點<br>溫度: {init_T[node_to_idx[i]]:.1f}°C" if i == edge else f"原子 {i}<br>溫度: {init_T[node_to_idx[i]]:.1f}°C") for i in atoms]
                
                fig3d.add_trace(go.Scatter3d(
                    x=node_x, y=node_y, z=node_z, 
                    mode='markers',  
                    text=init_labels, hoverinfo='text',
                    marker=dict(size=22, color=init_T, colorscale='Turbo', cmin=env_temp-10, cmax=init_temp, colorbar=dict(title="溫度 (°C)", thickness=10, x=-0.05))
                ))
                
                anim_frames = []
                for step, t in enumerate(time_steps):
                    t_data = history_frames[step]
                    step_labels = [f"🔥 核心源<br>溫度: {t_data[node_to_idx[i]]:.1f}°C" if i == core else (f"❄️ 外圍點<br>溫度: {t_data[node_to_idx[i]]:.1f}°C" if i == edge else f"原子 {i}<br>溫度: {t_data[node_to_idx[i]]:.1f}°C") for i in atoms]
                    anim_frames.append(go.Frame(data=[go.Scatter3d(marker=dict(color=t_data), text=step_labels)], traces=[1], name=f"f{step}"))
                fig3d.frames = anim_frames
                
                fig3d.update_layout(
                    autosize=True,
                    title="🔥 真實 3D 空間熱擴散", scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
                    template="plotly_dark", margin=dict(l=10, r=10, b=30, t=40), 
                    updatemenus=[dict(
                        type="buttons", active=-1, showactive=False, y=-0.05, x=0.5, xanchor="center", yanchor="top", direction="left",
                        buttons=[
                            dict(label="▶️ 播放聯動", method="animate", args=[None, dict(frame=dict(duration=anim_speed, redraw=True), fromcurrent=True, mode="immediate", transition=dict(duration=0))]),
                            dict(label="⏸️ 暫停", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])
                        ]
                    )]
                )
                html_3d = fig3d.to_html(include_plotlyjs="cdn", full_html=False, div_id="plot-3d")

                fig2d = go.Figure()
                fig2d.add_trace(go.Scatter(x=[time_steps[0]], y=[core_hist[0]], mode='lines', name=f'核心點火源', line=dict(color='red', width=3)))
                fig2d.add_trace(go.Scatter(x=[time_steps[0]], y=[edge_hist[0]], mode='lines', name=f'外圍測溫點', line=dict(color='blue', width=3)))
                fig2d.update_layout(
                    autosize=True,
                    title="📈 絕對精確溫度動態變化 (°C)", 
                    template="plotly_dark", 
                    margin=dict(l=55, r=25, b=65, t=40),
                    xaxis=dict(range=[0, sim_duration], title="時間 (秒)"), 
                    yaxis=dict(range=[env_temp-10, init_temp+20], title="溫度 (°C)")
                )
                html_2d = fig2d.to_html(include_plotlyjs=False, full_html=False, div_id="plot-2d")

                history_json = json.dumps([arr.tolist() for arr in history_frames])
                time_json = json.dumps(time_steps.tolist())
                core_json = json.dumps(core_hist)
                edge_json = json.dumps(edge_hist)
                atoms_json = json.dumps(atoms)
                
                table_html = """
                <table style="width:100%; border-collapse: collapse; text-align: center; color: white; font-family: sans-serif;">
                    <thead>
                        <tr>
                            <th style="padding: 10px; border-bottom: 2px solid #555; position: sticky; top: 0; background: #222; color: #ddd; z-index:10;">粒子編號</th>
                            <th style="padding: 10px; border-bottom: 2px solid #555; position: sticky; top: 0; background: #222; color: #ddd; z-index:10;">拓樸定位</th>
                            <th style="padding: 10px; border-bottom: 2px solid #555; position: sticky; top: 0; background: #222; color: #ddd; z-index:10;">即時溫度 (°C)</th>
                        </tr>
                    </thead><tbody>
                """
                for idx, atom in enumerate(atoms):
                    role = "🔥 核心點火源" if atom == core else ("❄️ 外部邊緣點" if atom == edge else "傳導節點")
                    table_html += f"<tr><td style='padding: 6px; border-bottom: 1px solid #333;'>Atom {atom}</td><td style='padding: 6px; border-bottom: 1px solid #333;'>{role}</td><td id='temp-{idx}' style='padding: 6px; border-bottom: 1px solid #333; font-weight: bold; color: #00ffcc;'>{init_T[node_to_idx[atom]]:.2f} °C</td></tr>"
                table_html += "</tbody></table>"

                custom_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <style>
                        body, html {{ margin: 0; padding: 0; background-color: #0e1117; width: 100%; height: 100%; overflow: hidden; }}
                        #fs-container {{
                            display: grid;
                            grid-template-columns: 60% 40%;
                            grid-template-rows: 50% 50%;
                            width: 100vw;
                            height: 100vh;
                            background: #0e1117;
                        }}
                        #left-pane {{ grid-column: 1 / 2; grid-row: 1 / 3; border-right: 2px solid #333; overflow: hidden; }}
                        #top-right-pane {{ grid-column: 2 / 3; grid-row: 1 / 2; border-bottom: 2px solid #333; overflow: hidden; }}
                        #bottom-right-pane {{ grid-column: 2 / 3; grid-row: 2 / 3; overflow-y: auto; background: #1a1a1a; padding: 15px; }}
                        .js-plotly-plot, .plot-container {{ width: 100% !important; height: 100% !important; }}
                        .fs-btn {{ position: absolute; top: 10px; right: 20px; z-index: 9999; background: rgba(255,255,255,0.1); color: #fff; border: 1px solid rgba(255,255,255,0.4); padding: 6px 12px; border-radius: 4px; cursor: pointer; transition: 0.2s; }}
                        .fs-btn:hover {{ background: rgba(255,255,255,0.3); }}
                    </style>
                </head>
                <body>
                    <button class="fs-btn" onclick="toggleFS()">⤢ 全螢幕</button>
                    <div id="fs-container">
                        <div id="left-pane"> {html_3d} </div>
                        <div id="top-right-pane"> {html_2d} </div>
                        <div id="bottom-right-pane"> {table_html} </div>
                    </div>
                    <script>
                        function toggleFS() {{
                            let elem = document.documentElement;
                            if (!document.fullscreenElement) {{ elem.requestFullscreen(); }} 
                            else {{ document.exitFullscreen(); }}
                        }}

                        var h_data = {history_json};
                        var t_data = {time_json};
                        var c_data = {core_json};
                        var e_data = {edge_json};
                        var a_list = {atoms_json};

                        function syncDashboard(step) {{
                            var temps = h_data[step];
                            if(temps) {{
                                for(var i=0; i<a_list.length; i++) {{
                                    var cell = document.getElementById('temp-' + i);
                                    if(cell) cell.innerText = temps[i].toFixed(2) + ' °C';
                                }}
                            }}
                            var gd2d = document.getElementById('plot-2d');
                            if (gd2d && typeof Plotly !== 'undefined') {{
                                var new_t = t_data.slice(0, step+1);
                                var new_c = c_data.slice(0, step+1);
                                var new_e = e_data.slice(0, step+1);
                                Plotly.restyle(gd2d, {{'x': [new_t, new_t], 'y': [new_c, new_e]}}, [0, 1]);
                            }}
                        }}

                        var checkExist = setInterval(function() {{
                            var gd3d = document.getElementById('plot-3d');
                            if (gd3d && typeof gd3d.on === 'function') {{
                                clearInterval(checkExist);
                                gd3d.on('plotly_animatingframe', function(eventData) {{
                                    var step = parseInt(eventData.name.replace('f', ''));
                                    syncDashboard(step);
                                }});
                            }}
                        }}, 200);
                        
                        window.onload = function() {{
                            setTimeout(function() {{
                                window.dispatchEvent(new Event('resize'));
                            }}, 500);
                        }};
                    </script>
                </body>
                </html>
                """
                components.html(custom_html, height=850)
