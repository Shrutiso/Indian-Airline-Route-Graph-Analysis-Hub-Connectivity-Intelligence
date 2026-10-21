# Apache Spark GraphX: Indian Airline Route Graph Analysis

An end-to-end Big Data Graph Processing and Network Analytics project utilizing **Apache Spark GraphX** and **Scala** to analyze India's domestic airline route network, calculate degree centralities, rank hubs using PageRank, uncover unserved isolated airports, and visualize the route graph topology.

---

## 1. Project Overview & Technology Stack

| Component | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Programming Language** | Scala | 2.12.18 | Core functional programming language |
| **Graph Processing Framework** | Apache Spark GraphX | 3.5.7 | Distributed graph construction, degree calculations, PageRank |
| **Data Processing** | Spark SQL / DataFrames | 3.5.7 | High-performance CSV parsing, filtering, and metric joins |
| **Build Tool** | sbt (Scala Build Tool) | 1.9.9 | Dependency resolution and JAR packaging |
| **Datasets** | Hugging Face + Master Registry | 2018–2025 | Cleaned domestic flight schedules & Indian airport master list |
| **Visualization** | Interactive D3.js + Gephi | HTML5 / CSV | Hub connectivity charts, force-directed route topology |

---

## 2. Directory Structure

```text
IndianAirlineGraphX/
├── build.sbt                                 # sbt build definition with Spark 3.5.7 & GraphX dependencies
├── sbt.bat                                   # One-click Windows runner script
├── project/
│   └── build.properties                      # Specifies sbt version (1.9.9)
├── data/
│   ├── Air-Clean.csv                         # Indian Domestic Airline dataset (33,734 flight schedules)
│   └── indian_airports_master.csv            # Master registry of 153 Indian airports (active + isolated)
├── src/
│   └── main/
│       └── scala/
│           └── AirlineRouteAnalysis.scala    # Core Spark GraphX Scala application
├── output/
│   ├── airport_analysis.csv                  # Complete metric table (degrees, centralities, PageRank)
│   ├── graph_nodes.csv                       # Node table for Gephi and web visualization
│   ├── graph_edges.csv                       # Directed edge table (origin -> destination)
│   ├── isolated_airports.csv                 # Detailed list of 43 zero-connection airports
│   └── hub_network_chart.html                # Interactive force-directed hub connectivity dashboard
└── scripts/
    ├── generate_master_airports.py           # Script that compiles the 153-airport master registry
    ├── eda_analysis.py                       # Exploratory Data Analysis (EDA) on datasets
    └── build_visualization.py                # Compiles output tables into interactive HTML graph
```

---

## 3. Step-by-Step Exploratory Data Analysis (EDA)

### Dataset 1: Flight Route Schedules (`data/Air-Clean.csv`)
* **Total Records:** 33,734 commercial flight schedules.
* **Coverage Period:** 2018 – 2025 domestic flights within India.
* **Attributes:** `airline`, `flightNumber`, `origin`, `destination`, `daysOfWeek`, `scheduledDepartureTime`, `scheduledArrivalTime`, `timezone`, `validFrom`, `validTo`, `lastUpdated`.
* **Airlines Represented:** IndiGo (56.5%), SpiceJet (10.0%), Air India (9.2%), GoAir (5.5%), Air India Express (5.3%), Vistara (4.5%), Alliance Air (4.4%), etc.
* **Directed Route Corridors:** **1,423 distinct directed pairs** (e.g., Delhi $\rightarrow$ Mumbai, Srinagar $\rightarrow$ Delhi).
* **Active Airport Locations in Routes:** **110 commercial cities/airports**.

#### Top 5 Most Frequent Flight Corridors in India
1. **Srinagar $\rightarrow$ Delhi:** 470 schedules
2. **Mumbai $\rightarrow$ Delhi:** 354 schedules
3. **Jammu $\rightarrow$ Srinagar:** 336 schedules
4. **Jammu $\rightarrow$ Delhi:** 298 schedules
5. **Delhi $\rightarrow$ Mumbai:** 285 schedules

---

### Dataset 2: Master Indian Airport Registry (`data/indian_airports_master.csv`)
#### Resolving Limitation 7: Discovering Truly Isolated Airports
* **The Problem:** In standard route datasets, only airports that appear as an `origin` or `destination` are recorded. If a graph is built strictly from route records, an airport with **zero** recorded flights will never exist as a vertex in the graph.
* **The Solution:** We compiled an authentic master registry of **153 Indian civil airports, defense civil enclaves, regional airstrips, and island bases** from official DGCA and AAI records:
  * **110 Active Commercial Airports:** (Delhi, Mumbai, Bengaluru, Hyderabad, Kolkata, Chennai, etc.)
  * **43 Genuinely Isolated / Unserved Airstrips:** (Car Nicobar, Campbell Bay, Utkela, Jeypore, Gondia, Solapur, Burnpur, Purnea, Behala, Mechuka, Tuting, Ziro, etc.)
* **The Result in GraphX:** By passing this complete registry into the GraphX vertex set, Spark calculates in-degree = 0, out-degree = 0, total connections = 0, and degree centrality = 0.0 for all 43 isolated airstrips.

---

## 4. Key Analytical Findings & GraphX Metrics

### A. Top 10 Busiest Indian Airport Hubs by Degree Centrality
Degree Centrality measures an airport's direct connectivity across the nation:
$$\text{Degree Centrality} = \frac{\text{In-Degree} + \text{Out-Degree}}{2 \times (|V| - 1)}$$

| Rank | Location | IATA | In-Degree (Inbound) | Out-Degree (Outbound) | Total Connections | Degree Centrality |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Delhi** | DEL | 77 | 79 | **156** | **0.5132** |
| **2** | **Bengaluru** | BLR | 71 | 71 | **142** | **0.4671** |
| **3** | **Hyderabad** | HYD | 68 | 68 | **136** | **0.4474** |
| **4** | **Mumbai** | BOM | 68 | 68 | **136** | **0.4474** |
| **5** | **Kolkata** | CCU | 54 | 54 | **108** | **0.3553** |
| **6** | **Ahmedabad** | AMD | 51 | 52 | **103** | **0.3388** |
| **7** | **Chennai** | MAA | 45 | 45 | **90** | **0.2961** |
| **8** | **Jaipur** | JAI | 36 | 40 | **76** | **0.2500** |
| **9** | **Pune** | PNQ | 38 | 38 | **76** | **0.2500** |
| **10** | **Guwahati** | GAU | 31 | 31 | **62** | **0.2039** |

* **Observation:** Delhi (Indira Gandhi International Airport) is India's foremost mega-hub, directly connected to over 51% of all registered airports in India. Bengaluru, Hyderabad, and Mumbai form the primary southern and western transit backbones.

---

### B. Top 10 Most Influential Airports by PageRank
PageRank evaluates not merely the quantity of direct connections, but the **quality and prestige** of those connections (an airport ranks higher if it is linked to other major hubs):

| Rank | Location | Total Connections | PageRank Score | Hub Significance |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **Delhi** | 156 | **7.0548** | Primary National Gateway Hub |
| **2** | **Bengaluru** | 142 | **6.3515** | Southern Tech & Aviation Hub |
| **3** | **Mumbai** | 136 | **6.0183** | Western Financial Hub |
| **4** | **Hyderabad** | 136 | **5.8948** | Major South-Central Transit Core |
| **5** | **Kolkata** | 108 | **5.2368** | Eastern India & Northeast Gateway |
| **6** | **Ahmedabad** | 103 | **4.5630** | Western Commercial Hub |
| **7** | **Chennai** | 90 | **3.8021** | South-Eastern Gateway |
| **8** | **Guwahati** | 62 | **3.2430** | Northeast Regional Connector |
| **9** | **Pune** | 76 | **3.2341** | High-density Intra-state Hub |
| **10** | **Jaipur** | 76 | **3.1471** | Northern Tourism & Business Hub |

---

### C. Least-Connected Active Regional Airports (Total Connections > 0)
These airports have scheduled flights in the dataset, but cater to specialized or nascent regional routes (e.g., UDAN regional connectivity scheme):

| Location | In-Degree | Out-Degree | Total Connections | Operational Context |
| :--- | :---: | :---: | :---: | :--- |
| **Ludhiana** | 1 | 0 | **1** | Sahnewal Airport (Regional feeder) |
| **Pathankot** | 1 | 0 | **1** | Civil Enclave (Defense/feeder) |
| **Bidar** | 1 | 0 | **1** | UDAN feeder airport in Karnataka |
| **Cooch-Behar** | 1 | 1 | **2** | North Bengal regional connection |
| **Adampur** | 2 | 0 | **2** | Doaba region civil enclave |
| **Kushinagar** | 3 | 0 | **3** | International Buddhist tourism enclave |
| **Pondicherry** | 2 | 2 | **4** | Regional leisure route (Bengaluru/Hyderabad) |
| **Bilaspur** | 2 | 2 | **4** | Bilasa Devi Kevat Airport (UDAN route) |
| **Tuticorin** | 2 | 2 | **4** | Coastal Tamil Nadu regional feeder |
| **Bathinda** | 2 | 2 | **4** | Malwa region feeder |

---

### D. Genuinely Isolated Airports (Total Connections = 0)
Identified by fusing the master airport database with GraphX. These airports represent unserved regional airstrips, military air bases without commercial flights in this dataset, and remote island stations:

| Location | State / UT | Category / Description | Degree Centrality |
| :--- | :--- | :--- | :---: |
| **Car Nicobar** | Andaman & Nicobar | Island defense and civil airstrip | 0.0 |
| **Campbell Bay** | Andaman & Nicobar | Southernmost naval air station (INS Baaz) | 0.0 |
| **Utkela** | Odisha | Kalahandi regional airstrip | 0.0 |
| **Jeypore** | Odisha | Koraput regional airstrip | 0.0 |
| **Gondia** | Maharashtra | Birsi Airport / National Flying Training Institute | 0.0 |
| **Solapur** | Maharashtra | Inactive commercial airport | 0.0 |
| **Akola** | Maharashtra | Non-scheduled regional aerodrome | 0.0 |
| **Amravati** | Maharashtra | Belora airstrip (under development) | 0.0 |
| **Burnpur** | West Bengal | IISCO / Asansol private airstrip | 0.0 |
| **Behala** | West Bengal | Kolkata flying club / training base | 0.0 |
| **Purnea** | Bihar | Air Force base / proposed civil enclave | 0.0 |
| **Muzaffarpur** | Bihar | Non-operational regional airstrip | 0.0 |
| **Ziro** | Arunachal Pradesh | Advanced Landing Ground (ALG) | 0.0 |
| **Mechuka** | Arunachal Pradesh | Strategic Himalayan defense ALG | 0.0 |
| **Tuting** | Arunachal Pradesh | Frontier Himalayan landing strip | 0.0 |
| **Walong** | Arunachal Pradesh | Easternmost landing strip | 0.0 |
| **Khandwa** | Madhya Pradesh | Non-operational central airstrip | 0.0 |
| **Warangal** | Telangana | Mamnoor historic airstrip | 0.0 |
| **Vellore** | Tamil Nadu | Abdullapuram regional airstrip | 0.0 |
| **Kavaratti** | Lakshadweep | Island administrative heliport | 0.0 |

*(Full 43 isolated airports are exported to `output/isolated_airports.csv`)*.

---

## 5. Hub Connectivity Visualization Guide

### Method 1: Instant Browser Hub Connectivity Chart (No Installation Required)
We generated a standalone interactive dashboard at **`output/hub_network_chart.html`**.
1. Double click or open [`output/hub_network_chart.html`](file:///c:/Users/Shruti%20Somvanshi/Desktop/Scala%20Project/output/hub_network_chart.html) in Chrome, Edge, or Firefox.
2. **Features available:**
   * Dynamic Force-Directed Simulation of all 153 airports and 1,423 directed routes.
   * Nodes sized by PageRank and colored by geographical zones (North, West, South, East, Northeast, Central, Islands, and Isolated).
   * Hover over any node to inspect In-Degree, Out-Degree, Total Routes, and highlight incoming/outgoing flights.
   * Filter presets: "View All (153)", "Top Mega Hubs (PR > 3.0)", "Active Commercial (110)", or "Isolated Only (43)".
   * Airport search bar with instant focus and zoom.

---

### Method 2: Visualizing with Gephi (Professional Network Analysis)
Follow these 3 simple steps to generate the hub connectivity chart in **Gephi**:

#### Step 1: Import Nodes Table
1. Open Gephi and select **New Project**.
2. Click **Data Laboratory** tab $\rightarrow$ **Import Spreadsheet**.
3. Select `output/graph_nodes.csv` (or browse into `output/graph_nodes/part-00000...csv`).
4. Set **As table:** `Nodes table`.
5. Ensure `id` is mapped to **ID** and `label` is mapped to **Label**.
6. Click **Next** $\rightarrow$ **Finish** (Select "Append to existing workspace").

#### Step 2: Import Edges Table
1. In Data Laboratory, click **Import Spreadsheet** again.
2. Select `output/graph_edges.csv` (or browse into `output/graph_edges/part-00000...csv`).
3. Set **As table:** `Edges table`.
4. Ensure `source` is mapped to **Source** and `target` to **Target**.
5. Set **Graph Type:** `Directed`.
6. Click **Finish** $\rightarrow$ select **Append to existing workspace**.

#### Step 3: Generate the Hub Connectivity Layout
1. Switch to the **Overview** tab.
2. In the **Layout** panel on the left, choose **ForceAtlas 2**.
3. Settings to adjust:
   * **Scaling:** Set between `5.0` and `10.0`.
   * Check the box for **Prevent Overlap**.
4. Click **Run** and watch the hubs cluster into natural geographic and traffic centers, then click **Stop**.
5. In the **Appearance** panel (top left):
   * **Nodes $\rightarrow$ Size:** Choose **Ranking** $\rightarrow$ select `pagerank` or `total_connections` (Min size: 4, Max size: 35).
   * **Nodes $\rightarrow$ Color:** Choose **Ranking** or **Partition** to color by degree centrality or zone.
6. Enable node labels by clicking the black **T** icon at the bottom of the graph viewport.

---

## 6. How to Run the Program from Scratch

### Prerequisites
* **Java:** JDK 8 (already installed and verified at `C:\jdk1.8.0_202`)
* **Hadoop Winutils:** Configured at `C:\hadoop`

### Running the Project
Open PowerShell or Command Prompt in the project folder:
```powershell
# 1. Package the Scala application
.\sbt.bat package

# 2. Run the GraphX application
.\sbt.bat "run"
```

The application will read the datasets, execute GraphX degree centrality and PageRank, print the full tables to stdout, and export clean CSV tables to `output/`.
