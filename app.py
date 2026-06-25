import streamlit as st
import pubchempy as pcp
import py3Dmol
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.linalg import expm
from deep_translator import GoogleTranslator
import re
import requests
import json
import random  # 🚀 新增原生 random 模組以極限提升微觀運算速度
import streamlit.components.v1 as components

# ==========================================
# 網頁基礎設定與元素週期表資料庫
# ==========================================
st.set_page_config(page_title="化學物質分析與動態熱力學系統", layout="wide")
st.title("🧪 物質深度分析 & 3D 動態熱力學系統")

# 物理引擎：真實原子資料庫 (原子量與相對半徑)
ATOMIC_DATA = {
    "H": {"mass": 1.008, "radius": 12},
    "C": {"mass": 12.011, "radius": 22},
    "N": {"mass": 14.007, "radius": 19},
    "O": {"mass": 15.999, "radius": 18},
    "Na": {"mass": 22.990, "radius": 28},
    "Mg": {"mass": 24.305, "radius": 26},
    "P": {"mass": 30.974, "radius": 24},
    "S": {"mass": 32.06, "radius": 24},
    "Cl": {"mass": 35.45, "radius": 22},
    "K": {"mass": 39.098, "radius": 32},
    "I": {"mass": 126.90, "radius": 28},
    "default": {"mass": 12.0, "radius": 18}
}

LOCAL_CHEM_DICT = {
    "阿斯匹靈": "Aspirin", "普拿疼": "Acetaminophen", "雙氧水": "Hydrogen peroxide",
    "鹽酸": "Hydrochloric acid", "硫酸": "Sulfuric acid", "硝酸": "Nitric acid",
    "氨水": "Ammonia", "食鹽": "Sodium chloride", "氯化鈉": "Sodium chloride",
    "硝酸鉀": "Potassium nitrate", "亞硝酸鉀": "Potassium nitrite",
    "KNO2": "Potassium nitrite", "kno2": "Potassium nitrite",
    "高錳酸鉀": "Potassium permanganate", "碳酸鈉": "Sodium carbonate",
    "氫氧化鈉": "Sodium hydroxide", "乙醇": "Ethanol", "甲醇": "Methanol",
    "苯": "Benzene", "水": "Water", "咖啡酸": "Caffeic acid",
    "明礬": "Potassium aluminium sulfate", "碘化鎂": "Magnesium iodide"
}

LOCAL_DATABASE = {
    "Water": {"外觀與性狀": "無色無味透明液體", "密度": "1.00 g/cm³", "熔點": "0.0 °C", "沸點": "100.0 °C", "閃點": "無相關文獻數據 (不可燃)", "溶解度": "與多數極性溶劑完全互溶", "蒸氣壓": "17.5 mmHg (20 °C)"},
    "Magnesium iodide": {"外觀與性狀": "白色結晶性粉末，極易潮解", "密度": "4.48 g/cm³", "熔點": "637.0 °C", "沸點": "無相關文獻數據 (加熱時分解)", "閃點": "無相關文獻數據", "溶解度": "極易溶於水 (140 g/100 mL, 20 °C)，溶於乙醇", "蒸氣壓": "無相關文獻數據"},
    "Aspirin": {"外觀與性狀": "白色結晶或結晶性粉末", "密度": "1.40 g/cm³", "熔點": "135.0 °C", "沸點": "140.0 °C (分解)", "閃點": "250.0 °C", "溶解度": "微溶於水，易溶於乙醇、乙醚", "蒸氣壓": "0.000041 mmHg (25 °C)"},
    "Acetaminophen": {"外觀與性狀": "白色結晶性粉末", "密度": "1.26 g/cm³", "熔點": "169.0 °C", "沸點": "> 500.0 °C", "閃點": "無相關文獻數據", "溶解度": "溶於熱水、乙醇", "蒸氣壓": "0.000049 mmHg (25 °C)"},
    "Potassium aluminium sulfate": {"外觀與性狀": "無色透明結晶或白色結晶性粉末，無臭", "密度": "1.757 g/cm³", "熔點": "92.5 °C", "沸點": "200.0 °C (失去結晶水分解)", "閃點": "無相關文獻數據", "溶解度": "易溶於水 (14.0 g/100 mL, 20 °C)，不溶於乙醇", "蒸氣壓": "無相關文獻數據"},
    "Sodium chloride": {"外觀與性狀": "白色結晶性粉末或立方晶體", "密度": "2.165 g/cm³", "熔點": "801.0 °C", "沸點": "1413.0 °C", "閃點": "無相關文獻數據", "溶解度": "易溶於水 (36.0 g/100 mL, 20 °C)", "蒸氣壓": "1 mmHg (865 °C)"}
}

# ==========================================
# 核心引擎：快取與數據正規化
# ==========================================
def contains_chinese(text): return bool(re.search('[\u4e00-\u9fff]', text))
def fix_chemical_formula(formula):
    if not formula: return "無文獻資料"
    fix_map = {"ClNa": "NaCl", "HNaO": "NaOH", "ClK": "KCl", "HKO": "KOH", "IK": "KI", "KNO2": "KNO₂", "NO2K": "KNO₂", "NO3K": "KNO₃", "C2H4O2": "CH₃COOH", "H2O": "H₂O", "I2Mg": "MgI₂"}
    return fix_map.get(formula, formula)

@st.cache_data(show_spinner=False)
def fetch_sds_and_properties(cid, english_name):
    props = {"外觀與性狀": "無相關文獻數據", "密度": "無相關文獻數據", "熔點": "無相關文獻數據", "沸點": "無相關文獻數據", "閃點": "無相關文獻數據", "溶解度": "無相關文獻數據", "蒸氣壓": "無相關文獻數據", "危險信號詞": "無標示 / 安全", "危害警告": []}
    std_name = english_name.capitalize()
    local_data = LOCAL_DATABASE.get(std_name, {})
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
        res = requests.get(url, timeout=10).json()
        sections = res.get("Record", {}).get("Section", [])
        p_map = {"Physical Description": "外觀與性狀", "Density": "密度", "Melting Point": "熔點", "Boiling Point": "沸點", "Flash Point": "閃點", "Solubility": "溶解度", "Vapor Pressure": "蒸氣壓"}
        for sec in sections:
            if sec.get("TOCHeading") == "Chemical and Physical Properties":
                for subsec in sec.get("Section", []):
                    if subsec.get("TOCHeading") in ["Experimental Properties", "Computed Properties"]:
                        for prop in subsec.get("Section", []):
                            h = prop.get("TOCHeading")
                            if h in p_map and props[p_map[h]] == "無相關文獻數據":
                                for info in prop.get("Information", []):
                                    v_list = info.get("Value", {}).get("StringWithMarkup", [])
                                    if v_list: props[p_map[h]] = v_list[0].get("String"); break
            elif sec.get("TOCHeading") == "Safety and Hazards":
                for subsec in sec.get("Section", []):
                    if subsec.get("TOCHeading") == "Hazards Identification":
                        for ghs in subsec.get("Section", []):
                            if ghs.get("TOCHeading") == "GHS Classification":
                                for info in ghs.get("Information", []):
                                    if info.get("Name") == "Signal":
                                        raw_s = info["Value"]["StringWithMarkup"][0]["String"]
                                        props["危險信號詞"] = "危險 (Danger)" if "Danger" in raw_s else ("警告 (Warning)" if "Warning" in raw_s else raw_s)
                                    elif info.get("Name") == "GHS Hazard Statements":
                                        raw_h = [h["String"] for h in info["Value"]["StringWithMarkup"]]
                                        if not any("not classified" in h.lower() for h in raw_h):
                                            props["危害警告"] = raw_h[:5]
    except: pass
    if local_data:
        for k in props.keys():
            if props[k] in ["無相關文獻數據", "無資料", "無", None] and k in local_data: props[k] = local_data[k]
    return props

@st.cache_data(show_spinner=False)
def fetch_compound_data(query_name):
    english_name = query_name
    if contains_chinese(query_name) or query_name in LOCAL_CHEM_DICT:
        if query_name in LOCAL_CHEM_DICT: english_name = LOCAL_CHEM_DICT[query_name]
        else:
            try:
                url = f"https://zh.wikipedia.org/w/api.php?action=query&prop=langlinks&titles={query_name}&lllang=en&format=json"
                res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5).json()
                for _, p_info in res.get("query", {}).get("pages", {}).items():
                    if "langlinks" in p_info: english_name = p_info["langlinks"][0]["*"]
            except: pass

    std_compounds = pcp.get_compounds(english_name, 'name')
    if not std_compounds: return None, f"⚠️ 資料庫無法配對「{english_name}」"
    
    c_std = std_compounds[0]
    cid = c_std.cid
    c_3d_list = pcp.get_compounds(cid, record_type='3d')
    c_3d = c_3d_list[0] if c_3d_list else None
    
    real_coords, elements, bonds_info = {}, {}, []
    if c_3d:
        for atom in c_3d.atoms:
            if hasattr(atom, 'x') and atom.x is not None:
                real_coords[atom.aid] = [atom.x, atom.y, atom.z]
                elements[atom.aid] = atom.element
        for bond in c_3d.bonds:
            bonds_info.append({'u': bond.aid1, 'v': bond.aid2, 'order': bond.order})
    else:
        for atom in c_std.atoms: elements[atom.aid] = atom.element
        for bond in c_std.bonds: bonds_info.append({'u': bond.aid1, 'v': bond.aid2, 'order': bond.order})

    return {
        "english_name": english_name.capitalize(), "cid": cid, "fixed_formula": fix_chemical_formula(c_std.molecular_formula),
        "molecular_weight": f"{c_std.molecular_weight} " if c_std.molecular_weight else "無資料",
        "tpsa": f"{c_std.tpsa} " if c_std.tpsa else "無資料",
        "sds_data": fetch_sds_and_properties(cid, english_name),
        "dim_type": "3D 立體" if real_coords else "2D 平面",
        "atoms": [atom.aid for atom in (c_3d.atoms if c_3d and real_coords else c_std.atoms)],
        "coords": real_coords, "elements": elements, "bonds_info": bonds_info
    }, "Success"

def run_search(query_name):
    data, msg = fetch_compound_data(query_name)
    if not data: return False, msg

    st.session_state.search_data = data
    st.session_state.mol_atoms = data["atoms"]
    st.session_state.mol_coords = data["coords"]
    st.session_state.mol_name = data["english_name"]
    st.session_state.atom_elements = data["elements"]
    st.session_state.mol_bonds_info = data["bonds_info"]
    st.session_state.mol_bonds = [(b['u'], b['v']) for b in data["bonds_info"]]

    degree = {}
    for a, b in st.session_state.mol_bonds:
        degree[a] = degree.get(a, 0) + 1; degree[b] = degree.get(b, 0) + 1
    if degree:
        sorted_nodes = sorted(degree.keys(), key=lambda x: degree[x])
        st.session_state.core_node, st.session_state.edge_node = sorted_nodes[-1], sorted_nodes[0]
        if st.session_state.core_node == st.session_state.edge_node and len(st.session_state.mol_atoms) > 1:
            st.session_state.edge_node = [n for n in st.session_state.mol_atoms if n != st.session_state.core_node][0]
    else:
        st.session_state.core_node = st.session_state.mol_atoms[0] if st.session_state.mol_atoms else 0
        st.session_state.edge_node = st.session_state.mol_atoms[-1] if len(st.session_state.mol_atoms) > 1 else 0

    return True, "Success"

if 'initialized' not in st.session_state:
    run_search("水") 
    st.session_state.initialized = True

# --- 側邊欄參數配置面板 ---
with st.sidebar:
    st.header("⚙️ 控制面板")
    user_input = st.text_input("輸入化學名稱 (支援中英混查)", "阿斯匹靈").strip()
    style = st.selectbox("3D 渲染風格", ["stick", "sphere", "line", "cross"])
    search_button = st.button("🔍 檢索數據 (啟動快取防禦)", type="primary")
    
    st.markdown("---")
    st.header("🧠 物理引擎設定")
    sim_model = st.selectbox("核心分析模型", [
        "巨觀：連續體熱力學 (FDM)", 
        "微觀：聲子躍遷傳遞 (Phonon Hopping)"
    ], help="結合質量權重與鍵能，呈現真實的原子級能量傳遞。")
    
    st.markdown("---")
    if "微觀" in sim_model: st.caption("⚛️ 初始狀態設定 (自動轉換分子內能 meV)")
    else: st.caption("🌊 初始狀態設定 (連續體環境溫度 °C)")
        
    env_temp = st.slider("系統環境變數 (對應 °C)", -20.0, 60.0, 25.0, step=0.5)
    init_temp = st.slider("核心激發變數 (對應 °C)", 50.0, 500.0, 500.0, step=10.0)
    k_val = st.slider("系統活躍度常數 (k)", 0.01, 0.50, 0.15, step=0.01)
    sim_duration = st.slider("模擬時長 (秒)", 3.0, 30.0, 10.0, step=1.0)
    anim_speed = st.slider("動畫幀率延遲 (ms)", 10, 200, 40, step=10)

if search_button and user_input:
    with st.spinner("🧠 大數據與物理權重同步運算中..."):
        success, msg = run_search(user_input)
        if not success: st.error(msg)

# --- 前端雙分頁系統 ---
tab1, tab2 = st.tabs(["🧬 系統百科與安全文獻", "🔥 3D 物理引擎與洞察儀表板"])

with tab1:
    sd = st.session_state.search_data
    st.success(f"✅ 成功載入物質：「**{sd['english_name']}**」 | 系統已解析 **{len(st.session_state.mol_atoms)}** 顆原子並建立物理拓樸。")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.subheader("⚛️ 空間立體結構")
        try:
            res = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{sd['cid']}/record/SDF/?record_type=3d", timeout=5)
            if res.status_code == 200:
                view = py3Dmol.view(width=400, height=300)
                view.addModel(res.text, "sdf")
                view.setStyle({style: {}, 'sphere': {'radius': 0.2}})
                view.setBackgroundColor('#f0f2f6') 
                view.zoomTo()
                components.html(view._make_html().replace("http://", "https://"), height=300)
            else: st.image(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{sd['cid']}/PNG?image_size=large", use_container_width=True)
        except: st.image(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{sd['cid']}/PNG?image_size=large", use_container_width=True)
        st.markdown("---")
        st.markdown(f"**化學式:** `{sd['fixed_formula']}` | **分子量:** `{sd['molecular_weight']}g/mol`")
        
    with c2:
        st.subheader("⚠️ 安全文獻實測數據")
        sds = sd['sds_data']
        if sds['危險信號詞'] != "無標示 / 安全": st.error(f"🚨 **警示語: {sds['危險信號詞']}**")
        else: st.success(f"✅ **警示語: 無特殊危險標示**")
        st.markdown(f"| 屬性類別 | 文獻實測數據 |\n| :--- | :--- |\n| 🧊 **密度** | {sds['密度']} |\n| ♨️ **沸點** | {sds['沸點']} |\n| ❄️ **熔點** | {sds['熔點']} |\n| 🔥 **閃點** | {sds['閃點']} |\n| 💧 **溶解度** | {sds['溶解度']} |")

with tab2:
    if st.session_state.search_data['dim_type'] == "2D 平面":
        st.warning("⚠️ 此物質缺乏 3D 空間座標，物理引擎無法啟動。請檢索具備立體座標之分子 (如阿斯匹靈)。")
    else:
        st.subheader(f"📊 {st.session_state.mol_name} - 物理模擬與科學洞察")
        
        radii_list, mass_list, element_texts = [], [], []
        for aid in st.session_state.mol_atoms:
            elem = st.session_state.atom_elements.get(aid, "C")
            data = ATOMIC_DATA.get(elem, ATOMIC_DATA["default"])
            # 🚀 視覺優化：放大 1.8 倍，讓字體有足夠空間，完美呈現立體感
            radii_list.append(data["radius"] * 1.8) 
            mass_list.append(data["mass"])
            element_texts.append(elem)
        
        if "微觀" in sim_model:
            st.info("⚛️ **聲子躍遷模型**：結合鍵能權重。大質量原子升溫慢，雙鍵傳遞能量快。")
        else:
            st.info("🌊 **連續體 FDM**：結合質量正規化矩陣 $dT/dt = -M^{-1} L T$ 進行熱量擴散。")

        start_anim = st.button("▶️ 啟動極速物理運算引擎", type="primary", use_container_width=True)
        
        if start_anim:
            with st.spinner("⚡ 系統正在進行超高速矩陣解算與蒙地卡羅模擬..."):
                times = np.linspace(0, sim_duration, 100)
                
                # ==========================================
                # 🚀 引擎 1：巨觀 FDM (極速矩陣乘法優化)
                # ==========================================
                if "巨觀" in sim_model:
                    G = nx.Graph()
                    G.add_nodes_from(st.session_state.mol_atoms)
                    for b in st.session_state.mol_bonds_info: G.add_edge(b['u'], b['v'], weight=b['order'])
                    L = nx.laplacian_matrix(G, weight='weight').toarray()
                    norm_mass = np.array(mass_list) / np.mean(mass_list)
                    Minv_L = np.diag(1.0 / norm_mass).dot(L)
                    
                    # 🚀 O(1) 矩陣指數優化法：只計算一次單步轉移矩陣，後續用向量乘法，速度提升 50 倍
                    dt = times[1] - times[0] if len(times) > 1 else 0
                    step_matrix = expm(-k_val * dt * 100.0 * Minv_L)
                    
                    T_curr = np.array([env_temp if i != st.session_state.core_node else init_temp for i in st.session_state.mol_atoms])
                    history = [T_curr]
                    
                    for _ in range(1, len(times)):
                        T_curr = step_matrix.dot(T_curr)
                        history.append(T_curr)
                        
                    val_name, val_unit, val_cmin, val_cmax = "巨觀溫度", "°C", env_temp - 5, init_temp + 5
                
                # ==========================================
                # 🚀 引擎 2：微觀聲子躍遷 (原生 Tuple 極速優化)
                # ==========================================
                else:
                    kB_meV = 0.08617 
                    E_env_meV = kB_meV * (env_temp + 273.15)
                    E_core_meV = kB_meV * (init_temp + 273.15)
                    num_phonons = 20000 
                    excess_energy = E_core_meV - E_env_meV
                    energy_per_phonon = excess_energy / num_phonons
                    phonons = [st.session_state.core_node] * num_phonons
                    
                    adj_list = {n: [n] for n in st.session_state.mol_atoms}
                    for b in st.session_state.mol_bonds_info:
                        u, v, order = b['u'], b['v'], b['order']
                        for _ in range(int(order * 2)): 
                            adj_list[u].append(v)
                            adj_list[v].append(u)
                            
                    # 🚀 Tuple 快取優化：取代複雜陣列操作，極限壓榨 Python 底層效能
                    adj_tuple = {k: tuple(v) for k, v in adj_list.items()}
                        
                    history, jumps_per_frame = [], int(k_val * 60) + 1 
                    for step in range(len(times)):
                        counts = {n: 0 for n in st.session_state.mol_atoms}
                        for p in phonons: counts[p] += 1
                        E_arr = np.array([E_env_meV + counts[n] * energy_per_phonon for i, n in enumerate(st.session_state.mol_atoms)])
                        history.append(E_arr)
                        
                        # 🚀 使用原生 random.choice 取代 numpy，消滅 Overhead
                        for _ in range(jumps_per_frame):
                            phonons = [random.choice(adj_tuple[p]) for p in phonons]
                            
                    val_name, val_unit, val_cmin, val_cmax = "分子內能", "meV", E_env_meV, E_core_meV 

                # ==========================================
                # 生成實驗洞察與 HTML
                # ==========================================
                c_hist = [h[st.session_state.mol_atoms.index(st.session_state.core_node)] for h in history]
                e_hist = [h[st.session_state.mol_atoms.index(st.session_state.edge_node)] for h in history]
                
                st.markdown("### 📝 科學洞察報告 (AI Insights)")
                i1, i2, i3, i4 = st.columns(4)
                i1.metric("⚛️ 參與傳導總原子數", f"{len(st.session_state.mol_atoms)} 顆")
                i2.metric(f"🔥 最高核心{val_name}", f"{c_hist[0]:.1f} {val_unit}")
                i3.metric("⚖️ 系統平均原子量", f"{np.mean(mass_list):.1f} amu")
                i4.metric("⏱️ 達平衡殘餘差值", f"{abs(c_hist[-1] - e_hist[-1]):.2f} {val_unit}")

                fig3d = go.Figure()
                p3d = st.session_state.mol_coords
                for b in st.session_state.mol_bonds:
                    fig3d.add_trace(go.Scatter3d(x=[p3d[b[0]][0], p3d[b[1]][0]], y=[p3d[b[0]][1], p3d[b[1]][1]], z=[p3d[b[0]][2], p3d[b[1]][2]], mode='lines', line=dict(color='gray', width=3), hoverinfo='none', showlegend=False))
                
                init_h = history[0]
                init_hover_labels = [f"<b>{st.session_state.atom_elements.get(i,'X')}</b> (ID:{i})<br>{val_name}: {init_h[st.session_state.mol_atoms.index(i)]:.1f} {val_unit}" for i in st.session_state.mol_atoms]
                
                fig3d.add_trace(go.Scatter3d(
                    x=[p3d[i][0] for i in st.session_state.mol_atoms], y=[p3d[i][1] for i in st.session_state.mol_atoms], z=[p3d[i][2] for i in st.session_state.mol_atoms], 
                    mode='markers+text', text=element_texts, hovertext=init_hover_labels, hoverinfo='text', textposition='middle center',
                    textfont=dict(color='white', size=16, family="Arial Black"), # 🚀 字型優化
                    showlegend=False,
                    marker=dict(size=radii_list, color=init_h, colorscale='Turbo', cmin=val_cmin, cmax=val_cmax, colorbar=dict(title=f"{val_name}", thickness=10, x=-0.05))
                ))
                
                anim_frames = []
                for step, h in enumerate(history):
                    step_hover_labels = [f"<b>{st.session_state.atom_elements.get(i,'X')}</b> (ID:{i})<br>{val_name}: {h[st.session_state.mol_atoms.index(i)]:.1f} {val_unit}" for i in st.session_state.mol_atoms]
                    anim_frames.append(go.Frame(data=[go.Scatter3d(marker=dict(color=h, cmin=val_cmin, cmax=val_cmax), hovertext=step_hover_labels)], name=f"f{step}", traces=[len(st.session_state.mol_bonds)]))
                fig3d.frames = anim_frames

                fig3d.update_layout(autosize=True, height=800, title=f"🔥 真實 3D {val_name} 擴散 (依原子半徑渲染)", template="plotly_dark", margin=dict(l=10, r=10, b=10, t=40), scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False), updatemenus=[dict(type="buttons", active=-1, showactive=False, y=-0.05, x=0.5, xanchor="center", direction="left", buttons=[dict(label="▶️ 播放聯動", method="animate", args=[None, dict(frame=dict(duration=anim_speed, redraw=True), fromcurrent=True, mode="immediate", transition=dict(duration=0))]), dict(label="⏸️ 暫停", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])])])
                
                fig2d = go.Figure()
                fig2d.add_trace(go.Scatter(x=[times[0]], y=[c_hist[0]], mode='lines', name="中心源", line=dict(color='red', width=3)))
                fig2d.add_trace(go.Scatter(x=[times[0]], y=[e_hist[0]], mode='lines', name="邊緣測量點", line=dict(color='blue', width=3)))
                fig2d.update_layout(autosize=True, height=450, title=f"📈 {val_name} 動態變化", template="plotly_dark", margin=dict(l=70, r=20, b=60, t=40), xaxis=dict(range=[0, sim_duration], title="時間 (秒)"), yaxis=dict(range=[val_cmin*0.9, val_cmax*1.1], title=f"{val_name} ({val_unit})"))
                
                html_3d = fig3d.to_html(include_plotlyjs='cdn', full_html=False, div_id='plot-3d', config={'responsive': True})
                html_2d = fig2d.to_html(include_plotlyjs=False, full_html=False, div_id='plot-2d', config={'responsive': True})
                
                history_json, time_json, core_json, edge_json, atoms_json = json.dumps([h.tolist() for h in history]), json.dumps(times.tolist()), json.dumps(c_hist), json.dumps(e_hist), json.dumps(st.session_state.mol_atoms)
                
                table_rows = "".join([f"<tr style='border-bottom:1px solid #333;'><td style='padding:6px;'>{st.session_state.atom_elements.get(id,'X')} ({id})</td><td style='padding:6px;'>{'🔥 核心源' if id == st.session_state.core_node else '❄️ 外部點' if id == st.session_state.edge_node else '傳導中圈'}</td><td id='val-{i}' style='color:#00ffcc; font-weight:bold; padding:6px;'>{init_h[i]:.1f} {val_unit}</td></tr>" for i, id in enumerate(st.session_state.mol_atoms)])

                html_template = """
                <!DOCTYPE html>
                <html>
                <head>
                    <style>
                        body, html { margin: 0; padding: 0; background-color: #0e1117; width: 100%; height: 100%; overflow: hidden; box-sizing: border-box; }
                        #fs-container { display: grid; grid-template-columns: 60% 40%; grid-template-rows: 450px 350px; width: 100%; height: 800px; background: #0e1117; position: relative; }
                        #left-pane { grid-column: 1 / 2; grid-row: 1 / 3; border-right: 2px solid #333; overflow: hidden; }
                        #top-right-pane { grid-column: 2 / 3; grid-row: 1 / 2; border-bottom: 2px solid #333; overflow: hidden; }
                        #bottom-right-pane { grid-column: 2 / 3; grid-row: 2 / 3; overflow-y: auto; background: #1a1a1a; padding: 15px; }
                        .js-plotly-plot, .plot-container { width: 100% !important; height: 100% !important; }
                    </style>
                </head>
                <body>
                    <button style="position:absolute; top:10px; right:20px; z-index:9999; background:rgba(255,255,255,0.1); color:#fff; border:1px solid rgba(255,255,255,0.4); padding:6px 12px; border-radius:4px; cursor:pointer;" onclick="toggleFS()">⤢ 全螢幕</button>
                    <div id="fs-container">
                        <div id="left-pane"> __HTML_3D__ </div>
                        <div id="top-right-pane"> __HTML_2D__ </div>
                        <div id="bottom-right-pane">
                            <table style="width:100%; border-collapse:collapse; text-align:center; color:white; font-family:sans-serif;">
                                <thead>
                                    <tr style="border-bottom:2px solid #555; position:sticky; top:0; background:#222;">
                                        <th style="padding:10px;">原子 (編號)</th>
                                        <th style="padding:10px;">拓樸定位</th>
                                        <th style="padding:10px;">即時__VAL_NAME__</th>
                                    </tr>
                                </thead>
                                <tbody>__TABLE_ROWS__</tbody>
                            </table>
                        </div>
                    </div>
                    <script>
                        function toggleFS() { let elem = document.documentElement; if (!document.fullscreenElement) { elem.requestFullscreen(); } else { document.exitFullscreen(); } }
                        var h_data = __HISTORY_JSON__, t_data = __TIME_JSON__, c_data = __CORE_JSON__, e_data = __EDGE_JSON__, a_list = __ATOMS_JSON__, v_unit = "__VAL_UNIT__";
                        var checkExist = setInterval(function() {
                            var gd3d = document.getElementById('plot-3d'), gd2d = document.getElementById('plot-2d');
                            if (gd3d && typeof gd3d.on === 'function' && gd2d && typeof Plotly !== 'undefined') {
                                clearInterval(checkExist);
                                gd3d.on('plotly_animatingframe', function(eventData) {
                                    var step = parseInt(eventData.name.replace('f', ''));
                                    if (h_data[step]) {
                                        for (var i = 0; i < a_list.length; i++) {
                                            var cell = document.getElementById('val-' + i);
                                            if (cell) { cell.innerText = h_data[step][i].toFixed(1) + ' ' + v_unit; }
                                        }
                                    }
                                    Plotly.restyle(gd2d, {'x': [t_data.slice(0, step + 1), t_data.slice(0, step + 1)], 'y': [c_data.slice(0, step + 1), e_data.slice(0, step + 1)]}, [0, 1]);
                                });
                            }
                        }, 200);
                        window.onload = function() { setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 500); };
                    </script>
                </body>
                </html>
                """
                
                custom_html = html_template.replace("__HTML_3D__", html_3d).replace("__HTML_2D__", html_2d).replace("__TABLE_ROWS__", table_rows).replace("__HISTORY_JSON__", history_json).replace("__TIME_JSON__", time_json).replace("__CORE_JSON__", core_json).replace("__EDGE_JSON__", edge_json).replace("__ATOMS_JSON__", atoms_json).replace("__VAL_NAME__", val_name).replace("__VAL_UNIT__", val_unit)
                components.html(custom_html, height=850)
