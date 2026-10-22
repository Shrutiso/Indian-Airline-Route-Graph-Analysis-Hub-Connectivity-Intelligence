# AeroGraphX India: Airline Route Graph Analysis & Hub Connectivity Intelligence

An end-to-end Big Data Graph Processing and Network Analytics project analyzing India's domestic airline route network. The platform models flight schedules as a **Directed Multigraph**, calculates **Degree Centrality**, evaluates network authority using **PageRank**, discovers unserved and isolated regional airstrips (**UDAN Scheme Analysis**), and presents interactive visualizations via **Streamlit** and **D3.js**.

---

## 1. Project Overview & Architecture

| Component | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Interactive Web Application** | Streamlit | 1.35.0 - 1.58.0 | Interactive executive dashboard, task walkthroughs, and viva summaries |
| **Graph Theory Engine** | NetworkX | 3.2.0+ | Directed graph construction, degree calculations, PageRank |
| **Data Processing** | Pandas & NumPy | 2.2.0+ / 1.26.0+ | Schedule aggregation, data filtering, and statistical analysis |
| **Data Visualizations** | Plotly & Matplotlib | 5.20.0+ / 3.8.0+ | Network topology diagrams, hub decay curves, interactive charts |
| **Big Data Engine (Optional)** | Apache Spark GraphX (Scala) | 3.5.7 / 2.12.18 | Distributed parallel graph processing via RDD message passing |
| **Testing Suite** | PyTest | 8.0.0+ | Automated unit tests verifying pipeline integrity and pathfinding |

---

## 2. Clean Modular Folder Structure

The project is structured so that every major operation has its own dedicated folder containing its analysis and visualization scripts:

```text
Scala Project/
├── app.py                                       # Main Streamlit web application
├── airline_route_analysis.py                    # Master end-to-end script executing all 6 tasks
├── requirements.txt                             # Python package dependencies
├── README.md                                    # Project documentation and viva guide
├── build.sbt                                    # Scala/Spark GraphX build configuration
├── sbt.bat                                      # One-click Windows SBT runner script
├── project/
│   └── build.properties                         # SBT version configuration (1.9.9)
├── Installation/
│   ├── setup_guide.md                           # Beginner-friendly step-by-step setup guide
│   └── requirements.txt                         # Dependencies reference copy
├── EDA/
│   ├── eda_analysis.py                          # Exploratory Data Analysis script
│   ├── eda_visualizations.py                    # Market share and top corridor charts
│   └── output_eda_summary.md                    # Generated EDA summary report
├── Task_1_Graph_Construction/
│   ├── task_1_analysis.py                       # Directed graph construction & topology validation
│   └── generate_master_airports.py              # Compiles authoritative 153-airport master list
├── Task_2_Direct_Connections/
│   ├── task_2_analysis.py                       # Computes In-Degree, Out-Degree, and Total Connections
│   └── task_2_visualization.py                  # Top 15 hubs horizontal bar chart
├── Task_3_Hub_Centrality/
│   ├── task_3_analysis.py                       # Computes Degree Centrality & identifies busiest hubs
│   └── task_3_visualization.py                  # Degree centrality decay curve & rankings
├── Task_4_Isolated_Airports/
│   ├── task_4_analysis.py                       # Identifies 43 isolated airports & least connected feeder strips
│   └── task_4_visualization.py                  # Distribution of unserved airstrips across zones
├── Task_5_PageRank_Analysis/
│   ├── task_5_analysis.py                       # Computes PageRank scores (d=0.85) for network transit authority
│   └── task_5_visualization.py                  # PageRank vs Degree Centrality comparison chart
├── Task_6_Network_Visualization/
│   ├── task_6_hub_connectivity_chart.py         # Matplotlib dual-panel hub connectivity chart
│   └── task_6_interactive_network.py            # D3.js interactive force-directed HTML graph builder
├── data/
│   ├── Air-Clean.csv                            # 33,734 commercial domestic flight schedules (2018–2025)
│   └── indian_airports_master.csv               # Master registry of 153 Indian airports (active + isolated)
├── output/
│   ├── airport_analysis.csv                     # Spark GraphX metric table
│   ├── graph_nodes.csv                          # Node table for D3.js and Gephi
│   ├── graph_edges.csv                          # Directed edge table (origin -> destination)
│   ├── isolated_airports.csv                    # Complete table of 43 zero-connection airports
│   ├── python_airport_analysis.csv              # Full Python network analysis table
│   ├── hub_connectivity_chart.png               # High-res dual-panel hub connectivity chart
│   └── hub_network_chart.html                   # Interactive standalone force-directed web dashboard
├── assets/
│   ├── indian_airline_connectivity.jpg          # Application hero image
│   └── hero_chart.png                           # Fallback chart asset
├── src/
│   ├── config.py                                # System settings, colors, and UI page definitions
│   ├── data_loader.py                           # Dataset loading and graph caching
│   ├── graph_analytics.py                       # Pathfinding and ego-network algorithms
│   ├── recommendations.py                       # Strategic aviation initiatives engine
│   ├── utils.py                                 # Formatting and logging helpers
│   ├── visualizations.py                        # Plotly chart generators
│   └── main/scala/
│       └── AirlineRouteAnalysis.scala           # Distributed Spark GraphX Scala source code
└── tests/
    └── test_analytics.py                        # Automated unit tests
```

---

## 3. Quick Start & Execution Commands

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501` to explore all tasks and the **"Major Findings & Summary"** section.

### Step 3: Run the Master Analytics Script (All 6 Tasks)
```powershell
python airline_route_analysis.py
```

### Step 4: Run Individual Tasks
```powershell
python EDA/eda_analysis.py
python Task_1_Graph_Construction/task_1_analysis.py
python Task_2_Direct_Connections/task_2_analysis.py
python Task_3_Hub_Centrality/task_3_analysis.py
python Task_4_Isolated_Airports/task_4_analysis.py
python Task_5_PageRank_Analysis/task_5_analysis.py
python Task_6_Network_Visualization/task_6_hub_connectivity_chart.py
```

### Step 5: Run Unit Tests
```powershell
pytest tests/
```

---

## 4. Key Analytical Findings & Viva Guide

### A. Top 10 Busiest Indian Airport Hubs by Degree Centrality
$$\text{Normalized Centrality} = \frac{\text{In-Degree} + \text{Out-Degree}}{2 \times (|V| - 1)}$$

| Rank | Location | IATA | In-Degree (Inbound) | Out-Degree (Outbound) | Total Connections | Degree Centrality | Network Role |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **Delhi** | DEL | 77 | 79 | **156** | **0.5132** | National Mega-Hub Gateway |
| **2** | **Bengaluru** | BLR | 71 | 71 | **142** | **0.4671** | Southern Tech & Aviation Backbone |
| **3** | **Hyderabad** | HYD | 68 | 68 | **136** | **0.4474** | South-Central Transit Core |
| **4** | **Mumbai** | BOM | 68 | 68 | **136** | **0.4474** | Western Commercial Hub |
| **5** | **Kolkata** | CCU | 54 | 54 | **108** | **0.3553** | Eastern India & Northeast Gateway |
| **6** | **Ahmedabad** | AMD | 51 | 52 | **103** | **0.3388** | Western Commercial Hub |
| **7** | **Chennai** | MAA | 45 | 45 | **90** | **0.2961** | Southeastern Coastal Hub |
| **8** | **Jaipur** | JAI | 36 | 40 | **76** | **0.2500** | Northern Tourism Core |
| **9** | **Pune** | PNQ | 38 | 38 | **76** | **0.2500** | Western Industrial Hub |
| **10** | **Guwahati** | GAU | 31 | 31 | **62** | **0.2039** | Northeast Regional Connector |

### B. Top 5 Airports by PageRank (Transit Authority)
- **Delhi (DEL):** 7.0548 (Highest transit authority across India)
- **Bengaluru (BLR):** 6.3515
- **Mumbai (BOM):** 6.0183
- **Hyderabad (HYD):** 5.8948
- **Kolkata (CCU):** 5.2368

### C. The 43 Isolated Airports Discovery (UDAN Gap Analysis)
In standard flight schedule tables, an unserved airport never appears as an origin or destination. By combining the official **153-airport master list**, our analysis uncovered that **43 airports (28.1%) have 0 commercial flights**.
- **Remote Island Bases:** Car Nicobar, Campbell Bay (INS Baaz), Kavaratti.
- **Himalayan Strategic ALGs:** Mechuka, Tuting, Ziro, Walong in Arunachal Pradesh.
- **Unserved Regional Strips:** Utkela, Jeypore, Solapur, Gondia, Burnpur.
- **Least Connected Active Airports:** Bidar, Ludhiana, and Pathankot rely on only **1 direct connection**.

### D. Practical Policy Implications
1. **UDAN Viability Gap Funding (VGF):** Provides empirical justification to subsidize flights to the 43 isolated regional airstrips.
2. **Decongesting Mega-Hubs:** Highlights the critical necessity of secondary greenfield airports (Noida Jewar for Delhi, Navi Mumbai for Mumbai).
3. **Point-to-Point Tier-2 Routes:** Encourages airlines to open direct Tier-2 routes (e.g. Pune–Guwahati, Ahmedabad–Coimbatore) to bypass congested mega-hubs.
