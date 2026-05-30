import streamlit as st
import pubchempy as pcp
import py3Dmol
import networkx as nx
import plotly.graph_objects as go
import time
import pandas as pd
from deep_translator import GoogleTranslator
import re
import requests
import streamlit.components.v1 as components

# 網頁基礎設定 (必須在最第一行)
st.set_page_config(page_title="中文化學物質分析與動態熱力學系統", layout="wide")

st.title("🧪 物質深度分析 & 3D 動態熱力學系統")
st.markdown("搭載 **維基百科對接引擎** 與 **真實分子拓樸解析**，熱傳導模擬將完全依據真實化學鍵推演，並支援高幀率流暢動畫。")

# ==========================================
# 核心一：維基百科學術名詞對接引擎
# ==========================================
LOCAL_CHEM_DICT = {
    "阿斯匹靈": "Aspirin", "普拿疼": "Acetaminophen", "雙氧水": "Hydrogen peroxide",
    "鹽酸": "Hydrochloric acid", "硫酸": "Sulfuric acid", "硝酸": "Nitric acid",
    "氨水": "Ammonia", "食鹽": "Sodium chloride", "硝酸鉀": "Potassium nitrate",
    "高錳酸鉀": "Potassium permanganate", "碳酸鈉": "Sodium carbonate",
    "氫氧化鈉": "Sodium hydroxide", "乙酸": "Acetic acid", "冰醋酸": "Acetic acid",
    "乙醇": "Ethanol", "酒精": "Ethanol", "甲醇": "Methanol",
    "苯": "Benzene", "水": "Water"
}

def contains_chinese(text):
    return bool(re.search('[\u4e00-\u9fff]', text))

def translate_via_wikipedia(zh_name):
    try:
        url = f"https://zh.wikipedia.org/w/api.php?action=query&prop=langlinks&titles={zh_name}&lllang=en&format=json"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        res = requests.get(url, headers=headers, timeout=5).json()
        pages = res.get("query", {}).get("pages", {})
        for page_id, page_info in pages.items():
            if "langlinks" in page_info:
                return page_info["langlinks"][0]["*"] 
    except:
        pass
    return None

# ==========================================
# 核心二：全繁中翻譯與 SDS 爬蟲
# ==========================================
def safe_translate(text):
    if not text or text == "無相關文獻數據":
        return text
    try:
        return GoogleTranslator(source='en', target='zh-TW').translate(text)
    except:
        return text

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
                                if heading == "Physical Description": props["外觀與性狀"] = safe_translate(val)
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

# --- 側邊欄：全局參數設定面板 ---
with st.sidebar:
    st.header("⚙️ 全局參數設定面板")
    st.subheader("🔬 1. 物質百科檢索")
    user_input = st.text_input("輸入中文試劑或藥品名稱", "乙醇").strip()
    style = st.selectbox("3D 顯示風格", ["stick", "sphere", "line", "cross"])
    search_button = st.button("🔍 執行數據檢索", type="primary")
    
    st.markdown("---")
    st.subheader("🔥 2. 熱傳導動態模擬參數")
    env_temp = st.slider("環境溫度設定 (°C)", min_value=-20.0, max_value=60.0, value=25.0, step=0.5)
    init_temp = st.slider("中心點火溫度 (°C)", min_value=50.0, max_value=500.0, value=300.0, step=10.0)
    k_val = st.slider("熱傳導係數 (k)", min_value=0.01, max_value=0.50, value=0.15, step=0.01)
    sim_speed = st.slider("時間流動速度 (幀延遲秒數)", min_value=0.01, max_value=0.50, value=0.05, step=0.01)

# ==========================================
# 初始化真實分子拓樸狀態機
# ==========================================
if 'mol_atoms' not in st.session_state:
    st.session_state.mol_atoms = list(range(10))
    st.session_state.mol_bonds = [(0,4), (0,5), (0,6), (1,4), (1,7), (1,8), (2,5), (2,7), (2,9), (3,6), (3,8), (3,9)]
    st.session_state.core_node = 0
    st.session_state.edge_node = 9
    st.session_state.mol_name = "預設測試結構"

# --- 雙分頁介面 ---
tab1, tab2 = st.tabs(["🧬 SDS 物質安全與化學百科", "🔥 真實分子拓樸熱傳導台"])

# ==========================================
# 分頁 1：化學百科與 SDS 危害報告
# ==========================================
with tab1:
    if search_button and user_input:
        with st.spinner("🧠 系統正在調閱學術詞典並解析真實分子拓樸..."):
            try:
                english_name = user_input
                translation_source = "原生輸入"
                
                if contains_chinese(user_input):
                    if user_input in LOCAL_CHEM_DICT:
                        english_name = LOCAL_CHEM_DICT[user_input]
                        translation_source = "本地字典"
                    else:
                        wiki_name = translate_via_wikipedia(user_input)
                        if wiki_name:
                            english_name = wiki_name
                            translation_source = "維基百科學術對接"
                        else:
                            translated = GoogleTranslator(source='auto', target='en').translate(user_input)
                            if contains_chinese(translated):
                                st.warning(f"⚠️ 無法自動辨識「{user_input}」，請嘗試輸入常見俗名或英文學名。")
                                st.stop()
                            else:
                                english_name = translated
                                translation_source = "Google 翻譯"

                compounds = pcp.get_compounds(english_name, 'name')
                if compounds:
                    c = compounds[0]
                    
                    # 🚀 解析真實原子與化學鍵
                    st.session_state.mol_atoms = [atom.aid for atom in c.atoms]
                    st.session_state.mol_bonds = [(bond.aid1, bond.aid2) for bond in c.bonds]
                    st.session_state.mol_name = english_name.capitalize()
                    
                    # 智能找出連結最多的核心原子 (點火點)
                    degree = {}
                    for a, b in st.session_state.mol_bonds:
                        degree[a] = degree.get(a, 0) + 1
                        degree[b] = degree.get(b, 0) + 1
                    
                    if degree:
                        st.session_state.core_node = max(degree, key=degree.get)
                        st.session_state.edge_node = min(degree, key=degree.get)
                    else:
                        st.session_state.core_node = st.session_state.mol_atoms[0] if st.session_state.mol_atoms else 0
                        st.session_state.edge_node = st.session_state.mol_atoms[-1] if st.session_state.mol_atoms else 0
                    
                    # 強制重置熱傳導狀態機
                    st.session_state.current_time = 0.0
                    st.session_state.particle_temps = {i: env_temp for i in st.session_state.mol_atoms}
                    st.session_state.particle_temps[st.session_state.core_node] = init_temp
                    
                    sds_data = fetch_sds_and_properties(c.cid)
                    st.success(f"✅ 檢索成功！物質映射：「**{english_name.capitalize()}**」 | 真實原子數: {len(st.session_state.mol_atoms)}")
                    
                    col_left, col_right = st.columns([1, 1.3])
                    with col_left:
                        st.subheader("⚛️ 3D 空間立體結構")
                        viewer = py3Dmol.view(query=f"cid:{c.cid}", width=450, height=350)
                        viewer.setStyle({style: {}})
                        viewer.setBackgroundColor('#f0f2f6')
                        viewer.zoomTo()
                        
                        # 🚀 原生 HTML 渲染，拔除 stmol 毒瘤
                        components.html(viewer._make_html(), height=350, width=450)
                        
                        st.markdown("---")
                        st.subheader("🧮 計算結構屬性")
                        st.markdown(f"""
                        * **化學式:** `{c.molecular_formula}`
                        * **分子量:** `{c.molecular_weight} g/mol`
                        * **TPSA (極性表面積):** `{c.tpsa} Å²`
                        * **氫鍵 (供體/受體):** `{c.h_bond_donor_count} / {c.h_bond_acceptor_count}`
                        """)

                    with col_right:
                        st.subheader("⚠️ SDS 物質安全與危害標示 (GHS)")
                        signal = sds_data["危險信號詞"]
                        
                        if "Danger" in signal or "危險" in signal: st.error(f"**🚨 警示語: {signal}**")
                        elif "Warning" in signal or "警告" in signal: st.warning(f"**⚠️ 警示語: {signal}**")
                        else: st.success("**✅ 警示語: 無特殊危險標示**")
                            
                        if sds_data["危害警告"]:
                            for h in sds_data["危害警告"]: st.caption(f"▪️ {h}")
                        else: st.caption("無查獲特定危害聲明紀錄。")

                        st.markdown("---")
                        st.subheader("🌡️ 實驗室文獻實測數據")
                        prop_md = f"""
                        | 屬性類別 | 文獻實測數值 (包含單位) |
                        | :--- | :--- |
                        | 🧊 **密度 (Density)** | {sds_data["密度"]} |
                        | ♨️ **沸點 (Boiling Point)** | {sds_data["沸點"]} |
                        | ❄️ **熔點 (Melting Point)** | {sds_data["熔點"]} |
                        | 🔥 **閃點 (Flash Point)** | {sds_data["閃點"]} |
                        | 💧 **溶解度 (Solubility)** | {sds_data["溶解度"]} |
                        | ☁️ **蒸氣壓 (Vapor Pressure)** | {sds_data["蒸氣壓"]} |
                        | 👁️ **外觀與性狀** | {sds_data["外觀與性狀"]} |
                        """
                        st.markdown(prop_md)
                else:
                    st.error(f"⚠️ 找不到與「{english_name}」匹配的物質。")
            except Exception as e:
                st.error(f"檢索發生錯誤：{e}")
    else:
        st.info("💡 請在左側輸入物質名稱，並按下「🔍 執行數據檢索」來啟動百科。")

# ==========================================
# 分頁 2：高幀率真實拓樸動態熱傳導台
# ==========================================
with tab2:
    st.subheader(f"🔥 {st.session_state.mol_name} - 真實分子熱能擴散監控")

    atoms = st.session_state.mol_atoms
    bonds = st.session_state.mol_bonds
    core = st.session_state.core_node
    edge = st.session_state.edge_node

    if 'current_time' not in st.session_state:
        st.session_state.current_time = 0.0
        st.session_state.particle_temps = {i: env_temp for i in atoms}
        st.session_state.particle_temps[core] = init_temp
        st.session_state.time_history = [0.0]
        st.session_state.core_history = [init_temp]
        st.session_state.edge_history = [env_temp]

    if st.session_state.current_time == 0.0:
        st.session_state.particle_temps = {i: env_temp for i in atoms}
        st.session_state.particle_temps[core] = init_temp
        st.session_state.core_history = [init_temp]
        st.session_state.edge_history = [env_temp]

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 4])
    with btn_col1:
        start_flow = st.button("▶️ 開始時間流動", type="primary", use_container_width=True)
    with btn_col2:
        reset_system = st.button("🔄 重置系統狀態", use_container_width=True)

    if reset_system:
        st.session_state.current_time = 0.0
        st.session_state.particle_temps = {i: env_temp for i in atoms}
        st.session_state.particle_temps[core] = init_temp
        st.session_state.time_history = [0.0]
        st.session_state.core_history = [init_temp]
        st.session_state.edge_history = [env_temp]
        st.rerun()

    col_visual, col_chart = st.columns([1.4, 1])
    with col_visual:
        plot_3d_placeholder = st.empty()
    with col_chart:
        plot_line_placeholder = st.empty()

    st.markdown("### 🔢 每個粒子的即時溫度數據面板")
    data_grid_placeholder = st.empty()

    G = nx.Graph()
    G.add_nodes_from(atoms)
    G.add_edges_from(bonds)
    
    if len(atoms) > 1:
        pos_3d = nx.spring_layout(G, dim=3, seed=42)
    else:
        pos_3d = {atoms[0]: [0, 0, 0]}
    
    edge_x, edge_y, edge_z = [], [], []
    for bond in G.edges():
        x0, y0, z0 = pos_3d[bond[0]]
        x1, y1, z1 = pos_3d[bond[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    def render_all_components():
        node_x = [pos_3d[i][0] for i in G.nodes()]
        node_y = [pos_3d[i][1] for i in G.nodes()]
        node_z = [pos_3d[i][2] for i in G.nodes()]
        node_colors = [st.session_state.particle_temps[i] for i in G.nodes()]
        
        node_labels = []
        for i in G.nodes():
            if i == core: prefix = "🔥 Core"
            elif i == edge: prefix = "❄️ Edge"
            else: prefix = f"Atom {i}"
            node_labels.append(f"{prefix}<br>{st.session_state.particle_temps[i]:.1f}°C")
        
        fig3d = go.Figure()
        fig3d.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color='gray', width=3), hoverinfo='none'))
        fig3d.add_trace(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            text=node_labels, textposition="top center", textfont=dict(size=11, color='white'),
            marker=dict(size=24, color=node_colors, colorscale='Turbo', cmin=-20, cmax=500,
                        colorbar=dict(title="溫度 (°C)", thickness=15), line=dict(width=2, color='white'))
        ))
        
        # 🚀 效能護城河：加入 uirevision='constant'
        fig3d.update_layout(title=f"⏳ 當前累積流動時間
