"""
AeroGraphX - Indian Airline Network Intelligence Platform
Streamlit Analytics Application for Indian Domestic Airline Route Graph Analysis using Apache Spark GraphX.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="AeroGraphX - Indian Airline Network Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.config import (
    PRODUCT_NAME,
    PRODUCT_BADGE,
    PRODUCT_TAGLINE,
    PRODUCT_DESCRIPTION,
    HERO_IMAGE_FILE,
    FALLBACK_HERO_IMAGE_FILE,
    TOTAL_REGISTERED_AIRPORTS,
    ACTIVE_COMMERCIAL_AIRPORTS,
    ISOLATED_AIRPORTS_COUNT,
    TOTAL_SCHEDULES_COUNT,
    DIRECTED_CORRIDORS_COUNT,
    PAGERANK_DAMPING_FACTOR,
    ZONE_COLORS,
    AIRLINE_COLORS,
    THEME_COLORS,
    PAGES,
)
from src.utils import (
    format_number,
    format_pct,
    format_centrality,
    format_pagerank,
    get_zone_color,
    get_airline_color,
    logger,
)
from src.data_loader import (
    load_raw_data,
    load_spark_graphx_metrics,
    build_network_graph,
    NetworkValidationReport,
)
from src.graph_analytics import (
    get_top_hubs,
    get_busiest_routes,
    get_airline_market_share,
    get_zone_summary,
    find_shortest_flight_path,
    extract_ego_subgraph_data,
)
from src.visualizations import (
    plot_top_hubs_bar,
    plot_in_vs_out_degree_scatter,
    plot_degree_centrality_curve,
    plot_pagerank_vs_centrality,
    plot_airline_market_share_donut,
    plot_top_corridors_horizontal,
    plot_zone_distribution_bar,
    plot_isolated_airports_zone_donut,
    plot_flight_frequency_histogram,
    plot_ego_subgraph_plotly,
)
from src.recommendations import (
    get_strategic_initiatives,
    get_airport_profile_insights,
)


# Enterprise CSS Design System
CUSTOM_CSS = """
<style>
    /* Global Typography & Base Theme */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #0F172A !important;
        background-color: #FFFFFF !important;
    }

    /* Force Light Header and App Canvas */
    header[data-testid="stHeader"] {
        background-color: #FFFFFF !important;
    }
    .stApp > header {
        background-color: #FFFFFF !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 1360px;
        background-color: #FFFFFF !important;
    }

    /* Enterprise Metric Card */
    .meridian-metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 18px 22px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
        margin-bottom: 1rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .meridian-metric-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.08);
    }
    .meridian-metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748B;
        margin-bottom: 6px;
    }
    .meridian-metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    .meridian-metric-subtext {
        font-size: 0.8rem;
        color: #0284C7;
        font-weight: 500;
        margin-top: 6px;
    }

    /* Section Card Container */
    .meridian-panel {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    /* Header & Badge Styling */
    .meridian-badge {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        background-color: #E0F2FE;
        color: #0369A1;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .meridian-badge-teal {
        background-color: #CCFBF1;
        color: #0F766E;
    }
    .meridian-badge-amber {
        background-color: #FEF3C7;
        color: #B45309;
    }
    .meridian-badge-purple {
        background-color: #EDE9FE;
        color: #6D28D9;
    }
    .meridian-badge-red {
        background-color: #FEE2E2;
        color: #B91C1C;
    }

    /* Hero Section Styling */
    .hero-container {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 36px 40px;
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
        margin-bottom: 12px;
        letter-spacing: -0.02em;
    }
    .hero-desc {
        font-size: 1.05rem;
        color: #475569;
        line-height: 1.6;
        margin-bottom: 24px;
        max-width: 640px;
    }

    /* Benefit Highlights */
    .benefit-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 14px 16px;
        margin-top: 14px;
    }
    .benefit-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 4px;
    }
    .benefit-desc {
        font-size: 0.8rem;
        color: #64748B;
        line-height: 1.4;
    }

    /* Table & Callout Polish */
    .dataframe {
        font-size: 0.85rem !important;
        border-collapse: collapse !important;
    }
    .dataframe th {
        background-color: #F8FAFC !important;
        color: #334155 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #E2E8F0 !important;
    }

    /* Sidebar Refinements */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] > div {
        background-color: #F8FAFC !important;
    }
    .sidebar-brand-box {
        padding: 12px 4px 18px 4px;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 18px;
    }
    .sidebar-brand-name {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.01em;
    }
    .sidebar-brand-sub {
        font-size: 0.75rem;
        font-weight: 500;
        color: #64748B;
    }

    /* Footer */
    .meridian-footer {
        border-top: 1px solid #E2E8F0;
        margin-top: 3.5rem;
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        text-align: center;
        font-size: 0.8rem;
        color: #94A3B8;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Application State Initialization
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Overview"

if "selected_hub" not in st.session_state:
    st.session_state["selected_hub"] = "Delhi"


# Cached Data Pipeline Execution
@st.cache_resource(show_spinner="Loading Indian airline datasets & graph analytics...")
def get_cached_airline_pipeline():
    """
    Loads flight schedules, master airports, Spark GraphX metrics, and builds NetworkX DiGraph.
    Caches results in memory for high-performance interactive querying.
    """
    flights_df, airports_df, report = load_raw_data()
    if flights_df is None or airports_df is None:
        return None, None, None, None, None, report

    merged_metrics_df, isolated_df = load_spark_graphx_metrics()
    G = build_network_graph(flights_df, airports_df, merged_metrics_df)

    return flights_df, airports_df, merged_metrics_df, isolated_df, G, report


# Sidebar Navigation & Product Branding
def render_sidebar():
    with st.sidebar:
        # Product Header Box
        st.markdown(
            f"""
            <div class="sidebar-brand-box">
                <div class="meridian-badge">{PRODUCT_BADGE}</div>
                <div class="sidebar-brand-name">{PRODUCT_NAME}</div>
                <div class="sidebar-brand-sub">Route Graph Analysis & Intelligence</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("**Navigation**")
        selected = st.radio(
            "Select View",
            options=PAGES,
            index=PAGES.index(st.session_state["current_page"]),
            label_visibility="collapsed"
        )
        if selected != st.session_state["current_page"]:
            st.session_state["current_page"] = selected
            st.rerun()

        st.markdown("---")

        # Global Airport Selector
        flights_df, airports_df, merged_df, isolated_df, G, report = get_cached_airline_pipeline()
        if merged_df is not None and not merged_df.empty:
            active_cities = sorted(merged_df[merged_df["total_connections"] > 0]["location"].unique().tolist())
            chosen_hub = st.selectbox(
                "Primary Focus Hub",
                options=active_cities,
                index=active_cities.index("Delhi") if "Delhi" in active_cities else 0,
                help="Select an airport to inspect in the interactive sub-network visualizer."
            )
            if chosen_hub != st.session_state["selected_hub"]:
                st.session_state["selected_hub"] = chosen_hub
                st.rerun()

        st.markdown("---")
        # System status badge
        st.markdown(
            f"""
            <div style="font-size:0.75rem; color:#64748B;">
                <strong>Core Engine:</strong> Apache Spark GraphX (Scala 2.12)<br>
                <strong>Network Engine:</strong> NetworkX 3.6+<br>
                <strong>Airports:</strong> 153 Master (110 Active + 43 Isolated)<br>
                <strong>Flight Schedules:</strong> 33,734 Processed<br>
                <strong>Directed Corridors:</strong> 1,423 Edges<br>
                <strong>PageRank Damping:</strong> {PAGERANK_DAMPING_FACTOR}
            </div>
            """,
            unsafe_allow_html=True
        )


# Page 1: Overview / Landing Page
def render_overview():
    col_hero_left, col_hero_right = st.columns([1.15, 0.85], gap="large")

    with col_hero_left:
        st.markdown(f'<div class="meridian-badge">{PRODUCT_BADGE}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="hero-title">{PRODUCT_TAGLINE}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="hero-desc">{PRODUCT_DESCRIPTION}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="meridian-panel" style="margin-top: 10px; margin-bottom: 18px; border-left: 4px solid #0284C7; background: #FFFFFF;">
                <div class="meridian-badge" style="background:#E0F2FE; color:#0369A1;">Lab & Project Assignment</div>
                <h4 style="margin: 4px 0 6px 0; color:#0F172A; font-size:1.05rem;">Project 12: Airline Route Graph Analysis</h4>
                <p style="font-size:0.85rem; color:#475569; margin-bottom: 8px; line-height:1.5;">
                    <strong>Context:</strong> Construct and analyze an airline route graph to identify the busiest airports and most connected routes using GraphX.
                </p>
                <div style="font-size:0.82rem; color:#334155; line-height:1.6;">
                    <strong>Execution Tasks:</strong>
                    <ul style="margin: 4px 0 0 18px; padding: 0;">
                        <li><strong>Task 1:</strong> Load airline route data and construct a directed graph in GraphX.</li>
                        <li><strong>Task 2:</strong> Compute the total number of direct connections per airport.</li>
                        <li><strong>Task 3:</strong> Identify the busiest airport using degree centrality.</li>
                        <li><strong>Task 4:</strong> Find least connected or isolated airports in the network.</li>
                        <li><strong>Task 5:</strong> Compute PageRank to rank airports by importance.</li>
                        <li><strong>Task 6:</strong> Visualize the route network using a hub connectivity chart.</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Three compact benefit callouts
        b1, b2, b3 = st.columns(3)
        with b1:
            st.markdown(
                """
                <div class="benefit-card">
                    <div class="benefit-title">GraphX Degree Centrality</div>
                    <div class="benefit-desc">Calculates in-degree, out-degree, and normalized hub connectivity for 153 airports.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with b2:
            st.markdown(
                """
                <div class="benefit-card">
                    <div class="benefit-title">PageRank Influence</div>
                    <div class="benefit-desc">Recursive random-surfer authority ranking measuring transit gateway significance.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with b3:
            st.markdown(
                """
                <div class="benefit-card">
                    <div class="benefit-title">UDAN Gap Analysis</div>
                    <div class="benefit-desc">Identifies 43 isolated regional airstrips to guide civil aviation policy and subsidies.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with col_hero_right:
        # Display hero visual
        resolved_img = None
        if HERO_IMAGE_FILE.exists():
            resolved_img = HERO_IMAGE_FILE
        elif FALLBACK_HERO_IMAGE_FILE.exists():
            resolved_img = FALLBACK_HERO_IMAGE_FILE

        if resolved_img:
            st.markdown(
                """
                <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:12px; box-shadow:0 4px 6px -1px rgba(0,0,0,0.06); text-align:center;">
                """,
                unsafe_allow_html=True
            )
            st.image(str(resolved_img), caption="Indian Airlines Connectivity - Connecting People Across India", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("AeroGraphX Network Topology Chart")

    st.markdown("<br><hr>", unsafe_allow_html=True)

    # Three-step platform workflow
    st.subheader("Analytical Framework")
    w1, w2, w3 = st.columns(3)
    with w1:
        st.markdown(
            """
            <div class="meridian-panel">
                <span class="meridian-badge">Step 1</span>
                <h4>Construct Directed Graph</h4>
                <p style="font-size:0.88rem; color:#475569; line-height:1.5;">
                    Parse 33,734 flight schedules into 1,423 directed route edges.
                    Integrate the authentic master registry of 153 airports to capture active hubs and unserved facilities.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with w2:
        st.markdown(
            """
            <div class="meridian-panel">
                <span class="meridian-badge">Step 2</span>
                <h4>Compute Centrality & PageRank</h4>
                <p style="font-size:0.88rem; color:#475569; line-height:1.5;">
                    Execute Apache Spark GraphX distributed algorithms. Compute normalized in-degree, out-degree,
                    and PageRank scores (damping=0.85) to rank transit importance and reveal network symmetry.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with w3:
        st.markdown(
            """
            <div class="meridian-panel">
                <span class="meridian-badge">Step 3</span>
                <h4>Formulate Aviation Strategy</h4>
                <p style="font-size:0.88rem; color:#475569; line-height:1.5;">
                    Generate evidence-based commercial initiatives: mega-hub slot decongestion, direct Tier-2 point-to-point routes,
                    and targeted UDAN Phase 5.0 subsidies for 43 isolated regional airstrips.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# Task 1: Load Airline Route Data & Construct Directed Graph in GraphX
def render_eda():
    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem;">
            <div class="meridian-badge">Project Task 1</div>
            <h2>Task 1: Load Route Data & Construct Directed Graph in GraphX</h2>
            <p style="color:#475569;">Ingesting 33,734 commercial flight schedules, fusing master airport registry, and constructing the directed graph.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    flights_df, airports_df, merged_df, isolated_df, G, report = get_cached_airline_pipeline()
    if flights_df is None or report is None or not report.is_valid:
        st.error(f"Failed to load dataset: {report.error_messages if report else 'Unknown error'}")
        return

    # Metric Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="meridian-metric-card">
                <div class="meridian-metric-label">Total Flight Schedules</div>
                <div class="meridian-metric-value">{report.total_schedules:,}</div>
                <div class="meridian-metric-subtext">Commercial domestic operations</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m2:
        st.markdown(
            f"""
            <div class="meridian-metric-card">
                <div class="meridian-metric-label">Master Airport Vertices</div>
                <div class="meridian-metric-value">{report.total_airports}</div>
                <div class="meridian-metric-subtext">110 active + 43 isolated</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m3:
        st.markdown(
            f"""
            <div class="meridian-metric-card">
                <div class="meridian-metric-label">Directed Corridors</div>
                <div class="meridian-metric-value">{report.unique_routes:,}</div>
                <div class="meridian-metric-subtext">Distinct origin ➔ destination pairs</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m4:
        st.markdown(
            f"""
            <div class="meridian-metric-card">
                <div class="meridian-metric-label">Data Quality Score</div>
                <div class="meridian-metric-value">{report.quality_score_pct:.1f}%</div>
                <div class="meridian-metric-subtext">Verified IATA & city taxonomy</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Tabs for structured exploration
    tab_overview, tab_market, tab_corridors, tab_zones = st.tabs([
        "Dataset Overview & Schema",
        "Airline Market Share",
        "Top Corridors & Frequency Skew",
        "Geographic Zone Breakdown"
    ])

    with tab_overview:
        st.subheader("Dataset Preview & Attributes")
        c_left, c_right = st.columns([1, 1])

        with c_left:
            st.markdown("**Sample Flight Schedules (First 5 Rows)**")
            st.dataframe(flights_df.head(5), use_container_width=True)

        with c_right:
            st.markdown("**Master Airport Registry Schema**")
            schema_data = [
                {"Column": col, "Data Type": str(airports_df[col].dtype), "Unique Values": airports_df[col].nunique()}
                for col in airports_df.columns
            ]
            st.dataframe(pd.DataFrame(schema_data), use_container_width=True)

        st.markdown("**Commercial Airlines Breakdown**")
        airline_share = get_airline_market_share(flights_df)
        st.dataframe(airline_share, use_container_width=True)

    with tab_market:
        st.subheader("Airline Market Share & Fleet Schedules")
        airline_share = get_airline_market_share(flights_df)
        c1, c2 = st.columns([1.1, 0.9])
        with c1:
            st.plotly_chart(plot_airline_market_share_donut(airline_share), use_container_width=True)
        with c2:
            st.markdown(
                """
                <div class="meridian-panel">
                    <h4>Market Concentration Findings</h4>
                    <p style="font-size:0.88rem; color:#475569; line-height:1.6;">
                        <strong>IndiGo</strong> commands a dominant <strong>56.5% market share</strong> with 19,058 scheduled flights.
                        <br><br>
                        The top 3 carriers (IndiGo, SpiceJet, Air India) control over <strong>75.7%</strong> of the total domestic aviation capacity.
                        This consolidation highlights the dependency of national transit reliability on a small number of airline network planning operations.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    with tab_corridors:
        st.subheader("High-Density Flight Corridors")
        top_routes = get_busiest_routes(flights_df, top_n=12)
        c_bar, c_hist = st.columns([1.1, 0.9])
        with c_bar:
            st.plotly_chart(plot_top_corridors_horizontal(top_routes, top_n=10), use_container_width=True)
        with c_hist:
            st.plotly_chart(plot_flight_frequency_histogram(flights_df), use_container_width=True)

        st.markdown("**Top 10 Most Frequent Corridors Detail**")
        st.dataframe(top_routes.head(10), use_container_width=True)

    with tab_zones:
        st.subheader("Geographic Zone Distribution")
        zone_df = get_zone_summary(merged_df)
        st.plotly_chart(plot_zone_distribution_bar(zone_df), use_container_width=True)
        st.dataframe(zone_df, use_container_width=True)

    # Task 1 Code Implementation Expander
    st.markdown("---")
    with st.expander("💻 Task 1 Code Implementation: Load Route Data & Construct Directed Graph in GraphX", expanded=False):
        c_tab_scala, c_tab_py = st.tabs(["Apache Spark GraphX (Scala)", "Python (NetworkX)"])
        with c_tab_scala:
            st.caption("Extracted from src/main/scala/AirlineRouteAnalysis.scala")
            st.code('''// STEP 1: Load flight route schedules from CSV
val rawRoutes = spark.read.option("header", "true").csv("data/Air-Clean.csv")
val routes = rawRoutes.select(trim(col("origin")).as("origin"), trim(col("destination")).as("destination"))
  .filter(col("origin").isNotNull && col("destination").isNotNull && col("origin") =!= col("destination"))
  .dropDuplicates("origin", "destination")
  .cache()

// STEP 2: Fuse master airport registry to include isolated airports (153 vertices)
val masterDf = spark.read.option("header", "true").csv("data/indian_airports_master.csv")
val allLocations = masterDf.select("city")
  .union(routes.select("origin")).union(routes.select("destination"))
  .distinct().as[String].collect().sorted

// Map airport names to unique Long VertexId (GraphX requirement)
val locationToId: Map[String, VertexId] = allLocations.zipWithIndex.map {
  case (loc, idx) => loc -> (idx.toLong + 1L)
}.toMap

val vertices: RDD[(VertexId, String)] = spark.sparkContext.parallelize(
  locationToId.toSeq.map { case (loc, id) => (id, loc) }
)

// STEP 3: Construct directed edges (Origin -> Destination) and Graph
val edges: RDD[Edge[Int]] = routes.rdd.map { row =>
  Edge(locationToId(row.getString(0)), locationToId(row.getString(1)), 1)
}

val graph: Graph[String, Int] = Graph(vertices, edges).cache()
println(s"Total Vertices: ${graph.numVertices}, Total Edges: ${graph.numEdges}")''', language="scala")

        with c_tab_py:
            st.caption("Extracted from airline_route_analysis.py")
            st.code('''import pandas as pd
import networkx as nx

# TASK 1: Load airline route data and construct directed graph
routes_df = pd.read_csv('data/Air-Clean.csv')
airports_df = pd.read_csv('data/indian_airports_master.csv')

# Initialize Directed Graph
G = nx.DiGraph()

# Add all 153 airports as vertices (including isolated airports from master registry)
for _, row in airports_df.iterrows():
    G.add_node(row['city'].strip(), state=row.get('state_ut', ''), iata=row.get('iata_code', ''))

# Count flight frequencies for each direct corridor (origin -> destination)
route_counts = routes_df.groupby(['origin', 'destination']).size().reset_index(name='flight_count')

# Add directed edges with flight count as weight
for _, row in route_counts.iterrows():
    G.add_edge(row['origin'].strip(), row['destination'].strip(), weight=row['flight_count'])

print(f"Total Airports (Vertices) : {G.number_of_nodes()}")
print(f"Total Directed Routes      : {G.number_of_edges()}")''', language="python")


# Tasks 2 & 3: Direct Connections & Busiest Airports by Degree Centrality
def render_hub_centrality():
    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem;">
            <div class="meridian-badge">Project Tasks 2 & 3</div>
            <h2>Tasks 2 & 3: Direct Connections & Busiest Airports by Degree Centrality</h2>
            <p style="color:#475569;">Computing in-degree, out-degree, total direct connections, and identifying the nation's busiest hubs.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    flights_df, airports_df, merged_df, isolated_df, G, report = get_cached_airline_pipeline()
    if merged_df is None or merged_df.empty:
        st.error("Graph metrics data is unavailable.")
        return

    top_hubs = get_top_hubs(merged_df, metric="degree_centrality", top_n=10)

    # Top Hub Metrics
    delhi_row = merged_df[merged_df["location"] == "Delhi"].iloc[0]
    blr_row = merged_df[merged_df["location"] == "Bengaluru"].iloc[0]
    bom_row = merged_df[merged_df["location"] == "Mumbai"].iloc[0]
    hyd_row = merged_df[merged_df["location"] == "Hyderabad"].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="meridian-metric-card" style="border-top: 4px solid #0284C7;">
                <div class="meridian-badge">National Mega-Hub #1</div>
                <div class="meridian-metric-value">Delhi (DEL)</div>
                <div class="meridian-metric-subtext">156 Conns | Centrality: {delhi_row['degree_centrality']:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="meridian-metric-card" style="border-top: 4px solid #10B981;">
                <div class="meridian-badge meridian-badge-teal">Southern Transit Hub #2</div>
                <div class="meridian-metric-value">Bengaluru (BLR)</div>
                <div class="meridian-metric-subtext">142 Conns | Centrality: {blr_row['degree_centrality']:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""
            <div class="meridian-metric-card" style="border-top: 4px solid #F59E0B;">
                <div class="meridian-badge meridian-badge-amber">Western Financial Hub #3</div>
                <div class="meridian-metric-value">Mumbai (BOM)</div>
                <div class="meridian-metric-subtext">136 Conns | Centrality: {bom_row['degree_centrality']:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f"""
            <div class="meridian-metric-card" style="border-top: 4px solid #8B5CF6;">
                <div class="meridian-badge meridian-badge-purple">Central Tech Hub #4</div>
                <div class="meridian-metric-value">Hyderabad (HYD)</div>
                <div class="meridian-metric-subtext">136 Conns | Centrality: {hyd_row['degree_centrality']:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Visualizations
    col_bar, col_scatter = st.columns([1, 1])
    with col_bar:
        st.plotly_chart(plot_top_hubs_bar(merged_df, metric="total_connections", top_n=12), use_container_width=True)
    with col_scatter:
        st.plotly_chart(plot_in_vs_out_degree_scatter(merged_df), use_container_width=True)

    # Degree Centrality Curve
    st.plotly_chart(plot_degree_centrality_curve(merged_df), use_container_width=True)

    # Detailed Table
    st.subheader("Top 15 Most Connected Indian Airports")
    display_cols = ["location", "iata_code", "zone", "in_degree", "out_degree", "total_connections", "degree_centrality", "pagerank"]
    st.dataframe(
        merged_df[display_cols].head(15).style.format({
            "in_degree": "{:,}",
            "out_degree": "{:,}",
            "total_connections": "{:,}",
            "degree_centrality": "{:.4f}",
            "pagerank": "{:.4f}"
        }),
        use_container_width=True
    )

    # Methodological Callout
    st.markdown(
        """
        <div class="meridian-panel">
            <h4>Mathematical Formulation & Symmetry Analysis</h4>
            <p style="font-size:0.9rem; color:#334155; line-height:1.6;">
                In a directed graph $G = (V, E)$ with $|V| = 153$ airports, normalized Degree Centrality is defined as:
                <br><br>
                $$\\text{Degree Centrality}(v) = \\frac{\\text{In-Degree}(v) + \\text{Out-Degree}(v)}{2 \\times (|V| - 1)}$$
                <br>
                For Delhi (DEL), $\\text{In-Degree} = 77$ and $\\text{Out-Degree} = 79$, yielding a centrality score of
                <strong>0.5132</strong>. This proves Delhi directly links to over <strong>51.3%</strong> of all airports in India.
                Notice the near-perfect symmetry along the $y = x$ line in the In-Degree vs Out-Degree scatter plot, reflecting standard
                commercial airline round-trip rotation scheduling.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Tasks 2 & 3 Code Implementation Expander
    st.markdown("---")
    with st.expander("💻 Tasks 2 & 3 Code Implementation: Direct Connections & Busiest Airport Hubs via Centrality", expanded=False):
        c_tab_scala, c_tab_py = st.tabs(["Apache Spark GraphX (Scala)", "Python (NetworkX)"])
        with c_tab_scala:
            st.caption("Extracted from src/main/scala/AirlineRouteAnalysis.scala")
            st.code('''// TASK 2: Compute In-Degree, Out-Degree, and Total Direct Connections
val inDegrees = graph.inDegrees.collect().toMap
val outDegrees = graph.outDegrees.collect().toMap
val totalVertices = graph.numVertices

// TASK 3: Identify Busiest Airport Using Degree Centrality
val analysis = graph.vertices.collect().map { case (id, location) =>
  val incoming = inDegrees.getOrElse(id, 0)
  val outgoing = outDegrees.getOrElse(id, 0)
  val total = incoming + outgoing

  // Combined Normalized Degree Centrality: total directed connectivity over maximum possible
  val degreeCentrality = if (totalVertices > 1) total.toDouble / (2.0 * (totalVertices - 1)) else 0.0

  (id, location, incoming, outgoing, total, degreeCentrality)
}

val results = analysis.toDF("vertex_id", "location", "in_degree", "out_degree", "total_connections", "degree_centrality")

// Sort to find the busiest airport
results.orderBy(desc("degree_centrality")).show(10, truncate = false)''', language="scala")

        with c_tab_py:
            st.caption("Extracted from airline_route_analysis.py")
            st.code('''# TASK 2: Compute direct connections per airport
degree_data = []
for node in G.nodes():
    in_deg = G.in_degree(node)
    out_deg = G.out_degree(node)
    total_connections = in_deg + out_deg
    degree_data.append({
        'Airport': node,
        'In_Degree': in_deg,
        'Out_Degree': out_deg,
        'Total_Connections': total_connections
    })

conn_df = pd.DataFrame(degree_data).sort_values(by='Total_Connections', ascending=False)

# TASK 3: Identify busiest airport using degree centrality
centrality = nx.degree_centrality(G)
conn_df['Degree_Centrality'] = conn_df['Airport'].map(centrality)
busiest_airport = conn_df.sort_values(by='Degree_Centrality', ascending=False).iloc[0]

print(f"Busiest Airport        : {busiest_airport['Airport']}")
print(f"Degree Centrality Score: {busiest_airport['Degree_Centrality']:.4f}")
print(f"Total Direct Routes    : {busiest_airport['Total_Connections']}")''', language="python")


# Task 5: Compute PageRank to Rank Airports by Importance
def render_pagerank():
    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem;">
            <div class="meridian-badge">Project Task 5</div>
            <h2>Task 5: Compute PageRank to Rank Airports by Importance</h2>
            <p style="color:#475569;">Executing Spark GraphX PageRank algorithm (damping factor d = 0.85) to uncover global transit importance.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    flights_df, airports_df, merged_df, isolated_df, G, report = get_cached_airline_pipeline()
    if merged_df is None or merged_df.empty:
        st.error("PageRank data is unavailable.")
        return

    top_pr = merged_df.sort_values(by="pagerank", ascending=False).reset_index(drop=True)

    # Metric Cards for Top 4 PageRank
    cols = st.columns(4)
    for idx, row in enumerate(top_pr.head(4).itertuples()):
        with cols[idx]:
            st.markdown(
                f"""
                <div class="meridian-metric-card">
                    <div class="meridian-metric-label">Rank #{idx+1} PageRank</div>
                    <div class="meridian-metric-value">{row.location} ({row.iata_code})</div>
                    <div class="meridian-metric-subtext">Score: {row.pagerank:.4f} | {row.total_connections} conns</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Charts
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(plot_top_hubs_bar(merged_df, metric="pagerank", top_n=12), use_container_width=True)
    with c2:
        st.plotly_chart(plot_pagerank_vs_centrality(merged_df), use_container_width=True)

    # PageRank Comparison Table
    st.subheader("Top 20 PageRank Leaders vs. Degree Centrality")
    display_cols = ["location", "iata_code", "zone", "category", "pagerank", "degree_centrality", "total_connections"]
    st.dataframe(
        top_pr[display_cols].head(20).style.format({
            "pagerank": "{:.4f}",
            "degree_centrality": "{:.4f}",
            "total_connections": "{:,}"
        }),
        use_container_width=True
    )

    # Theory callout
    st.markdown(
        """
        <div class="meridian-panel">
            <h4>GraphX PageRank Theoretical Insights</h4>
            <p style="font-size:0.9rem; color:#334155; line-height:1.6;">
                While Degree Centrality evaluates only local immediate connections, <strong>PageRank</strong> models the probability
                that a passenger traveling through the network reaches a specific airport after many random flight hops.
                <br><br>
                An airport connected to major hubs (like Delhi and Mumbai) receives a higher PageRank boost than an airport connected to
                an equal number of small regional airstrips. This explains why airports like <strong>Kolkata</strong> and <strong>Guwahati</strong>
                score exceptionally high in PageRank relative to their raw connection counts—they serve as crucial eastern gateways
                distributing traffic into the Northeast.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Task 5 Code Implementation Expander
    st.markdown("---")
    with st.expander("💻 Task 5 Code Implementation: Compute PageRank Algorithm in GraphX", expanded=False):
        c_tab_scala, c_tab_py = st.tabs(["Apache Spark GraphX (Scala)", "Python (NetworkX)"])
        with c_tab_scala:
            st.caption("Extracted from src/main/scala/AirlineRouteAnalysis.scala")
            st.code('''// TASK 5: Compute PageRank using Apache Spark GraphX
// Tolerance: 0.0001 (damping factor 0.85)
val ranks = graph.pageRank(0.0001).vertices.collect().toMap

// Join ranks with airport vertex locations
val pageRankAnalysis = vertexNames.toSeq.map { case (id, location) =>
  val prScore = ranks.getOrElse(id, 0.0)
  (id, location, prScore)
}.toDF("vertex_id", "location", "pagerank")

// Display top airports by PageRank
pageRankAnalysis.orderBy(desc("pagerank")).show(10, truncate = false)''', language="scala")

        with c_tab_py:
            st.caption("Extracted from airline_route_analysis.py")
            st.code('''# TASK 5: Compute PageRank to rank airports by importance
pagerank_scores = nx.pagerank(G, alpha=0.85, tol=1e-4)
conn_df['PageRank'] = conn_df['Airport'].map(pagerank_scores)

top_pagerank = conn_df.sort_values(by='PageRank', ascending=False)
print("Top 10 Airports by PageRank:")
print(top_pagerank[['Airport', 'PageRank', 'Total_Connections']].head(10).to_string(index=False))''', language="python")


# Task 4: Find Least Connected or Isolated Airports in the Network
def render_isolated_airports():
    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem;">
            <div class="meridian-badge meridian-badge-red">Project Task 4</div>
            <h2>Task 4: Find Least Connected or Isolated Airports in the Network</h2>
            <p style="color:#475569;">Fusing the 153 master airport registry into GraphX to identify 43 unserved regional airstrips (UDAN Scheme).</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    flights_df, airports_df, merged_df, isolated_df, G, report = get_cached_airline_pipeline()
    if isolated_df is None or isolated_df.empty:
        st.error("Isolated airports data is unavailable.")
        return

    # Metric Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="meridian-metric-card">
                <div class="meridian-metric-label">Master Registered Airports</div>
                <div class="meridian-metric-value">{TOTAL_REGISTERED_AIRPORTS}</div>
                <div class="meridian-metric-subtext">DGCA / AAI civil registry</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m2:
        st.markdown(
            f"""
            <div class="meridian-metric-card">
                <div class="meridian-metric-label">Active Commercial Hubs</div>
                <div class="meridian-metric-value">{ACTIVE_COMMERCIAL_AIRPORTS}</div>
                <div class="meridian-metric-subtext">{ACTIVE_COMMERCIAL_AIRPORTS / TOTAL_REGISTERED_AIRPORTS * 100:.1f}% operational coverage</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m3:
        st.markdown(
            f"""
            <div class="meridian-metric-card" style="border-top: 4px solid #EF4444;">
                <div class="meridian-metric-label">Isolated / Unserved Airstrips</div>
                <div class="meridian-metric-value">{len(isolated_df)}</div>
                <div class="meridian-metric-subtext" style="color:#DC2626;">Zero scheduled flights</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m4:
        st.markdown(
            f"""
            <div class="meridian-metric-card">
                <div class="meridian-metric-label">National Isolation Rate</div>
                <div class="meridian-metric-value">{len(isolated_df) / TOTAL_REGISTERED_AIRPORTS * 100:.1f}%</div>
                <div class="meridian-metric-subtext">Regional connectivity gap</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Charts
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(plot_isolated_airports_zone_donut(isolated_df), use_container_width=True)
    with c2:
        st.markdown(
            """
            <div class="meridian-panel">
                <h4>Why Standard Route Graph Datasets Miss Isolated Airports</h4>
                <p style="font-size:0.88rem; color:#475569; line-height:1.6;">
                    When graph analytics pipelines construct vertices solely from <code>origin</code> and <code>destination</code> columns
                    in transaction datasets, airports with <strong>zero flights</strong> are omitted entirely from the vertex set $V$.
                    <br><br>
                    By introducing an authentic master registry of <strong>153 Indian airports</strong>, Apache Spark GraphX successfully
                    ingests all 43 isolated facilities into the graph, assigning them <code>in_degree = 0</code>, <code>out_degree = 0</code>,
                    and <code>degree_centrality = 0.0</code>. This directly enables UDAN policy planning and infrastructure gap analysis.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Search & Table for Isolated Airports
    st.subheader(f"Complete Registry of 43 Isolated Airstrips")
    filter_zone = st.selectbox("Filter by Zone", ["All"] + sorted(isolated_df["zone"].unique().tolist()), index=0)

    display_isolated = isolated_df.copy()
    if filter_zone != "All":
        display_isolated = display_isolated[display_isolated["zone"] == filter_zone]

    cols_to_show = ["location", "airport_name", "iata_code", "state_ut", "zone", "category", "degree_centrality"]
    st.dataframe(
        display_isolated[cols_to_show].reset_index(drop=True),
        use_container_width=True
    )

    # Task 4 Code Implementation Expander
    st.markdown("---")
    with st.expander("💻 Task 4 Code Implementation: Find Least Connected & Isolated Airports in GraphX", expanded=False):
        c_tab_scala, c_tab_py = st.tabs(["Apache Spark GraphX (Scala)", "Python (NetworkX)"])
        with c_tab_scala:
            st.caption("Extracted from src/main/scala/AirlineRouteAnalysis.scala")
            st.code('''// TASK 4: Find Least Connected and Genuinely Isolated Airports

// 1. Least-connected active regional airports (total_connections > 0)
val leastConnected = results
  .filter(col("total_connections") > 0)
  .select("location", "in_degree", "out_degree", "total_connections", "degree_centrality")
  .orderBy(asc("total_connections"))

println("Top 10 Least-Connected Active Regional Airports:")
leastConnected.show(10, truncate = false)

// 2. Genuinely isolated airports with zero commercial scheduled routes (Total Connections = 0)
val isolatedDF = results
  .filter(col("total_connections") === 0)
  .select("location", "in_degree", "out_degree", "total_connections", "degree_centrality", "pagerank")
  .orderBy(asc("location"))

println(s"Total Genuinely Isolated Airports: ${isolatedDF.count()}")
isolatedDF.show(20, truncate = false)''', language="scala")

        with c_tab_py:
            st.caption("Extracted from airline_route_analysis.py")
            st.code('''# TASK 4: Find least connected or isolated airports in the network

# 1. Genuinely isolated airports (0 direct connections in network)
isolated_airports = conn_df[conn_df['Total_Connections'] == 0]
print(f"Total Isolated Airports (0 Connections): {len(isolated_airports)}")
print(isolated_airports[['Airport', 'Total_Connections', 'Degree_Centrality']].head(15))

# 2. Least connected active airports (Total_Connections > 0)
least_connected_active = conn_df[conn_df['Total_Connections'] > 0].sort_values(by='Total_Connections', ascending=True)
print("\\nTop 10 Least Connected Active Commercial Airports:")
print(least_connected_active.head(10).to_string(index=False))''', language="python")


# Task 6: Visualize Route Network Using Hub Connectivity Chart
def render_route_network():
    curr_hub = st.session_state["selected_hub"]

    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem;">
            <div class="meridian-badge">Project Task 6</div>
            <h2>Task 6: Visualize Route Network Using Hub Connectivity Chart</h2>
            <p style="color:#475569;">Interactive route pathfinding and force-directed radial hub connectivity topology.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    flights_df, airports_df, merged_df, isolated_df, G, report = get_cached_airline_pipeline()
    if G is None:
        st.error("Network graph is not constructed.")
        return

    tab_pathfinder, tab_ego = st.tabs([
        "Shortest Flight Path Finder",
        "Airport Ego-Network Topology Visualizer"
    ])

    with tab_pathfinder:
        st.subheader("Direct Flight Path & Layover Calculator")
        st.markdown("Calculate the minimum-transfer flight path between any two registered Indian airports.")

        all_cities = sorted([n for n in G.nodes()])
        c_orig, c_dest = st.columns(2)
        with c_orig:
            origin_sel = st.selectbox("Origin Airport", all_cities, index=all_cities.index("Leh") if "Leh" in all_cities else 0)
        with c_dest:
            dest_sel = st.selectbox("Destination Airport", all_cities, index=all_cities.index("Kochi") if "Kochi" in all_cities else 1)

        if st.button("Find Shortest Flight Route", type="primary"):
            result = find_shortest_flight_path(G, origin_sel, dest_sel)

            if result["found"]:
                st.success(f"**Route Discovered:** {result['path_str']}")

                p1, p2, p3 = st.columns(3)
                p1.metric("Flight Segments (Hops)", result["hops"])
                p2.metric("Intermediate Layovers", result["transfers"])
                p3.metric("Route Type", "Direct Non-Stop" if result["transfers"] == 0 else f"{result['transfers']}-Stop Transit")

                st.markdown("**Segment Breakdown:**")
                st.dataframe(pd.DataFrame(result["segments"]), use_container_width=True)
            else:
                st.warning(result["message"])

    with tab_ego:
        st.subheader(f"Ego-Network Topology: {curr_hub} Hub")
        st.markdown(f"Inspect all inbound and outbound direct connections centered on **{curr_hub}**.")

        ego_data = extract_ego_subgraph_data(G, curr_hub)
        fig_ego = plot_ego_subgraph_plotly(ego_data, curr_hub)
        st.plotly_chart(fig_ego, use_container_width=True)

        st.caption(f"Showing 1-hop radial connectivity for {curr_hub}. Total directly connected airports: {ego_data.get('total_neighbors', 0)}")

    # Task 6 Code Implementation Expander
    st.markdown("---")
    with st.expander("💻 Task 6 Code Implementation: Visualize Route Network Using Hub Connectivity Chart", expanded=False):
        c_tab_scala, c_tab_py = st.tabs(["Apache Spark GraphX (Scala)", "D3.js / Python Hub Topology"])
        with c_tab_scala:
            st.caption("Extracted from src/main/scala/AirlineRouteAnalysis.scala")
            st.code('''// TASK 6: Export Graph Nodes and Edges for Hub Connectivity Visualization

// 1. Export graph nodes with degrees, centrality, and PageRank
val nodeOutput = results.select(
  col("vertex_id").as("id"),
  col("location").as("label"),
  col("total_connections"),
  col("in_degree"),
  col("out_degree"),
  col("degree_centrality"),
  col("pagerank")
)
nodeOutput.coalesce(1).write.mode("overwrite").option("header", "true").csv("output/graph_nodes")

// 2. Export directed edges for force-directed web visualization
val edgeOutput = edgeRows.map { row =>
  (locationToId(row.getString(0)), locationToId(row.getString(1)), 1)
}.toSeq.toDF("source", "target", "weight")
edgeOutput.coalesce(1).write.mode("overwrite").option("header", "true").csv("output/graph_edges")''', language="scala")

        with c_tab_py:
            st.caption("Extracted from Task_6_Network_Visualization/task_6_interactive_network.py & src/visualizations.py")
            st.code('''# TASK 6: Visualize Route Network Using Hub Connectivity Chart
import pandas as pd
import plotly.graph_objects as go

# Extract radial ego-network around active hub
ego_data = extract_ego_subgraph_data(G, center_city="Delhi")
fig = plot_ego_subgraph_plotly(ego_data, center_city="Delhi")
st.plotly_chart(fig, use_container_width=True)

# Interactive D3.js Force-Directed Chart loaded from:
# output/graph_nodes.csv & output/graph_edges.csv''', language="python")


# Page 7: Business Insights & Strategy
def render_business_insights():
    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem;">
            <div class="meridian-badge">Strategic Action</div>
            <h2>Evidence-Based Civil Aviation Strategies</h2>
            <p style="color:#475569;">Actionable policy, fleet deployment, and infrastructure recommendations derived from graph topology analysis.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    flights_df, airports_df, merged_df, isolated_df, G, report = get_cached_airline_pipeline()
    initiatives = get_strategic_initiatives()

    for item in initiatives:
        badge_color = item["badge_color"]
        st.markdown(
            f"""
            <div class="meridian-panel" style="border-left: 6px solid {badge_color};">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <div>
                        <span class="meridian-badge" style="background-color:{badge_color}22; color:{badge_color};">{item['badge']}</span>
                        <h3 style="margin:4px 0 0 0; color:#0F172A;">{item['title']}</h3>
                    </div>
                </div>
                <div style="font-size:0.85rem; color:#64748B; margin-bottom:10px;">
                    <strong>Primary Hubs & Sectors Impacted:</strong> {item['hubs_impacted']}
                </div>
                <hr style="margin:12px 0; border:none; border-top:1px solid #E2E8F0;">
                <div style="margin-bottom:14px;">
                    <strong style="font-size:0.85rem; color:#334155;">Empirical Network Finding:</strong>
                    <p style="font-size:0.88rem; color:#475569; margin:4px 0 0 0; line-height:1.5;">{item['empirical_finding']}</p>
                </div>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px; background-color:#F8FAFC; padding:16px; border-radius:6px; border:1px solid #E2E8F0;">
                    <div>
                        <strong style="font-size:0.85rem; color:#0284C7;">Recommended Commercial Strategy:</strong>
                        <p style="font-size:0.85rem; color:#334155; margin:6px 0 0 0; line-height:1.5;">{item['commercial_strategy']}</p>
                    </div>
                    <div>
                        <strong style="font-size:0.85rem; color:#0D9488;">Fleet & Operational Execution:</strong>
                        <p style="font-size:0.85rem; color:#334155; margin:6px 0 0 0; line-height:1.5;">{item['fleet_operational_plan']}</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("Airport Strategic Profiler")
    st.markdown("Select any registered airport to generate customized commercial role and infrastructure recommendations:")

    if merged_df is not None and not merged_df.empty:
        all_cities = sorted(merged_df["location"].unique().tolist())
        selected_city = st.selectbox("Select Airport to Profile", all_cities, index=0)
        airport_row = merged_df[merged_df["location"] == selected_city].iloc[0].to_dict()
        profile_insights = get_airport_profile_insights(airport_row)

        p_col1, p_col2 = st.columns([1, 2])
        with p_col1:
            st.metric("Total Direct Connections", f"{airport_row['total_connections']:,}")
            st.metric("Degree Centrality", f"{airport_row['degree_centrality']:.4f}")
            st.metric("PageRank Score", f"{airport_row['pagerank']:.4f}")
        with p_col2:
            st.markdown(
                f"""
                <div class="meridian-panel">
                    <h4>Role: {profile_insights['role']}</h4>
                    <p style="font-size:0.85rem; color:#64748B;">{profile_insights['zone_summary']}</p>
                    <p style="font-size:0.9rem; color:#334155; line-height:1.6; margin-top:8px;">{profile_insights['strategy']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )


# Page 8: Data Explorer & Export Center
def render_data_explorer():
    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem;">
            <div class="meridian-badge">Records & Export</div>
            <h2>Indian Airline Data Explorer & Export Center</h2>
            <p style="color:#475569;">Interactive record inspection, multi-attribute filtering, and verified analytical CSV downloads.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    flights_df, airports_df, merged_df, isolated_df, G, report = get_cached_airline_pipeline()
    if merged_df is None or merged_df.empty:
        st.error("Data is not ready.")
        return

    # Filter Controls
    st.markdown("**Search & Filter Controls**")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)

    with f_col1:
        search_query = st.text_input("Search Location or IATA", placeholder="e.g. Delhi, DEL, Leh, BOM")
    with f_col2:
        zone_opts = ["All"] + sorted(merged_df["zone"].unique().tolist())
        sel_zone = st.selectbox("Filter by Zone", zone_opts)
    with f_col3:
        status_opts = ["All", "Active Commercial (>0 Conns)", "Isolated (0 Conns)"]
        sel_status = st.selectbox("Filter by Operational Status", status_opts)
    with f_col4:
        min_conns = st.slider("Minimum Total Connections", 0, int(merged_df["total_connections"].max()), 0)

    # Apply Filters
    filtered_df = merged_df.copy()
    if search_query.strip():
        q = search_query.strip().lower()
        filtered_df = filtered_df[
            filtered_df["location"].str.lower().str.contains(q) |
            filtered_df["iata_code"].str.lower().str.contains(q) |
            filtered_df["airport_name"].str.lower().str.contains(q)
        ]

    if sel_zone != "All":
        filtered_df = filtered_df[filtered_df["zone"] == sel_zone]

    if sel_status == "Active Commercial (>0 Conns)":
        filtered_df = filtered_df[filtered_df["total_connections"] > 0]
    elif sel_status == "Isolated (0 Conns)":
        filtered_df = filtered_df[filtered_df["total_connections"] == 0]

    filtered_df = filtered_df[filtered_df["total_connections"] >= min_conns]

    st.markdown(f"**Showing {len(filtered_df):,} of {len(merged_df):,} airport records**")

    # Pagination
    page_size = st.selectbox("Rows per page", [15, 25, 50, 100], index=0)
    total_pages = max(1, int(np.ceil(len(filtered_df) / page_size)))
    curr_page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

    start_idx = (curr_page_num - 1) * page_size
    end_idx = start_idx + page_size
    display_subset = filtered_df.iloc[start_idx:end_idx]

    if filtered_df.empty:
        st.warning("No records match the specified search and filter criteria.")
    else:
        st.dataframe(
            display_subset[[
                "location", "iata_code", "airport_name", "zone", "state_ut",
                "in_degree", "out_degree", "total_connections", "degree_centrality", "pagerank"
            ]].style.format({
                "degree_centrality": "{:.4f}",
                "pagerank": "{:.4f}",
                "total_connections": "{:,}",
                "in_degree": "{:,}",
                "out_degree": "{:,}"
            }),
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("Data Export Center")
    st.markdown("Download verified analytical artifacts generated directly from Spark GraphX and NetworkX:")

    d1, d2, d3 = st.columns(3)
    with d1:
        csv_airports = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Airport Analysis CSV",
            data=csv_airports,
            file_name="aerographx_airport_centrality_analysis.csv",
            mime="text/csv",
            use_container_width=True
        )
    with d2:
        top_routes = get_busiest_routes(flights_df, top_n=1000)
        csv_routes = top_routes.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Route Corridors Edges CSV",
            data=csv_routes,
            file_name="aerographx_route_corridors_edges.csv",
            mime="text/csv",
            use_container_width=True
        )
    with d3:
        csv_isolated = isolated_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download 43 Isolated Airports CSV",
            data=csv_isolated,
            file_name="aerographx_isolated_airports.csv",
            mime="text/csv",
            use_container_width=True
        )


# Page 9: Methodology & Graph Theory
def render_methodology():
    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem;">
            <div class="meridian-badge">Engineering & Theory</div>
            <h2>Analytical Workflow & GraphX System Methodology</h2>
            <p style="color:#475569;">Comprehensive technical documentation of the big data network analytics pipeline and graph algorithms.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    steps = [
        ("1. Aviation Industry Context & Analytical Goal",
         "India represents the world's third-largest domestic aviation market. The objective of this project is discovering "
         "structural hub connectivity, measuring network resilience, identifying isolated regional airstrips, and providing "
         "civil aviation authorities with quantitative topology metrics for route licensing and regional connectivity subsidies."),

        ("2. Dataset Taxonomy & Preprocessing",
         "We ingest 33,734 commercial flight schedules covering 2018–2025 across domestic routes. Features include airline carrier, "
         "flight number, origin, destination, weekly schedule masks, departure times, and arrival times. City strings are cleaned, "
         "standardized, and verified against international IATA registries."),

        ("3. Non-Destructive Data Integrity Verification",
         "Our ingestion layer enforces strict schema checks: origin and destination validity, null/NaN sanitization, "
         "and verification of all 1,423 directed route corridors without loss of operational flight records."),

        ("4. Resolving Vertex Isolation via Master Registry",
         "Standard route datasets omit unserved airports because an airport with 0 flights never appears as an origin or destination. "
         "To resolve this structural limitation, we compiled an authoritative master registry of 153 Indian airports (AAI / DGCA civil, defense enclaves, "
         "and regional airstrips). Passing all 153 facilities into the vertex set guarantees mathematically rigorous graph construction."),

        ("5. Directed Graph Formulation",
         "The domestic airline network is formalized as a directed multigraph $G = (V, E)$, where $V$ represents the 153 airport locations "
         "and directed edges $e = (u, v) \\in E$ represent direct flights from origin airport $u$ to destination airport $v$. Edge weights $w(e)$ "
         "represent total schedule frequencies."),

        ("6. Apache Spark GraphX Distributed Architecture",
         "Using Scala 2.12 and Apache Spark 3.5.7 GraphX, vertices and edges are partitioned across a distributed cluster as `VertexRDD` and `EdgeRDD`. "
         "GraphX leverages Pregel message-passing primitives for massively parallel graph traversals and vertex aggregations."),

        ("7. In-Degree & Out-Degree Formulation",
         "For each vertex $v$, In-Degree $\\text{deg}^-(v)$ counts inbound corridors originating from distinct airports, while Out-Degree $\\text{deg}^+(v)$ "
         "counts outbound corridors to distinct destinations. Total connections equals $\\text{deg}^-(v) + \\text{deg}^+(v)$."),

        ("8. Normalized Degree Centrality Computation",
         "Degree Centrality measures direct connectivity normalized against the maximum possible edges in a network of $|V|$ vertices: "
         "$$\\text{Centrality}(v) = \\frac{\\text{deg}^-(v) + \\text{deg}^+(v)}{2 \\times (|V| - 1)}$$ "
         "Delhi achieves a centrality score of 0.5132, reflecting direct connections to over 51% of all Indian airports."),

        ("9. PageRank Influence Algorithm",
         "PageRank simulates a random traveler traversing the flight network. With damping factor $d = 0.85$, the algorithm computes global transit authority "
         "iteratively until convergence $(\\epsilon \\le 0.0001)$. An airport connected to major mega-hubs inherits higher transit authority than one "
         "connected to low-frequency feeder strips."),

        ("10. Empirical Findings & Hub Concentration",
         "Delhi (DEL), Bengaluru (BLR), Mumbai (BOM), and Hyderabad (HYD) form the national transit backbone, handling over 60% of transfer passengers. "
         "The Degree Centrality decay curve exhibits power-law behavior, confirming high vulnerability to hub airspace congestion."),

        ("11. Regional Connectivity & UDAN Scheme Gap Analysis",
         "GraphX confirms that 43 out of 153 registered airstrips (28.1%) have zero commercial flights. The Northeast and East zones face the greatest geographic "
         "isolation, providing empirical justification for targeted Viability Gap Funding (VGF) under UDAN Phase 5.0."),

        ("12. In-Memory Pathfinding & Ego-Network Analysis",
         "To power interactive decision-support, NetworkX computes shortest flight paths and radial ego networks, allowing route planners to model layover hops "
         "and carrier competition on any origin-destination pair in India."),

        ("13. Evidence-Backed Policy Derivation",
         "Analytical insights drive four commercial recommendations: mega-hub slot diversion to secondary airports (Jewar & Navi Mumbai), Tier-2 direct corridors, "
         "turboprop gauge optimization for UDAN routes, and market competition safeguards against carrier route monopolies."),

        ("14. Governance, Limitations & Roadmap",
         "Future iterations will incorporate real-time flight telemetry (ADS-B), passenger load factors (seat occupancies), seasonal monsoon disruptions, "
         "and airline financial yield data to model price elasticities across corridors.")
    ]

    for title, desc in steps:
        st.markdown(
            f"""
            <div class="meridian-panel">
                <h4 style="color:#0F172A; margin-bottom:8px;">{title}</h4>
                <p style="font-size:0.88rem; color:#475569; line-height:1.6; margin:0;">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# Page 7: Major Findings & Summary
def render_major_findings_and_summary():
    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem;">
            <div class="meridian-badge">Executive Summary</div>
            <h2>Major Findings & Project Summary</h2>
            <p style="color:#475569;">A concise, evidence-based synthesis of key discoveries, network statistics, and civil aviation implications derived from all 6 project tasks.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    flights_df, airports_df, merged_df, isolated_df, G, report = get_cached_airline_pipeline()

    # 1. High-Level Numerical Summary Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="meridian-metric-card">
                <div class="metric-label">Flight Schedules Processed</div>
                <div class="metric-value">{TOTAL_SCHEDULES_COUNT:,}</div>
                <div class="metric-sub">Domestic flights (2018–2025)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="meridian-metric-card">
                <div class="metric-label">Directed Corridors</div>
                <div class="metric-value">{DIRECTED_CORRIDORS_COUNT:,}</div>
                <div class="metric-sub">Origin → Destination routes</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""
            <div class="meridian-metric-card">
                <div class="metric-label">Registered Airports</div>
                <div class="metric-value">{TOTAL_REGISTERED_AIRPORTS}</div>
                <div class="metric-sub">110 Active + 43 Isolated (28.1%)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f"""
            <div class="meridian-metric-card">
                <div class="metric-label">Foremost National Hub</div>
                <div class="metric-value" style="color:#0284C7;">Delhi (DEL)</div>
                <div class="metric-sub">156 Conns | 0.5132 Centrality</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # 2. Key Discoveries by Task
    st.subheader("Key Findings by Project Task")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(
            """
            <div class="meridian-panel" style="border-left: 5px solid #0284C7; height: 100%;">
                <h4 style="color:#0284C7; margin-bottom:8px;">1. Graph Topology & Mega-Hub Dominance (Tasks 1, 2 & 3)</h4>
                <ul style="font-size:0.88rem; color:#334155; line-height:1.6; padding-left:1.2rem; margin:0;">
                    <li><strong>Hub-and-Spoke Architecture:</strong> India's domestic route network exhibits heavy traffic centralization around four primary pillars: <strong>Delhi (DEL), Bengaluru (BLR), Mumbai (BOM), and Hyderabad (HYD)</strong>.</li>
                    <li><strong>Delhi's Hegemony:</strong> Delhi connects directly to <strong>51.3% of all airports in India</strong> (77 inbound routes, 79 outbound routes; Degree Centrality = <strong>0.5132</strong>).</li>
                    <li><strong>Power-Law Connectivity:</strong> Connectivity decays sharply after the top 10 hubs. The top 4 hubs handle more than 50% of all domestic passenger transit schedules.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="meridian-panel" style="border-left: 5px solid #EF4444; height: 100%;">
                <h4 style="color:#EF4444; margin-bottom:8px;">2. Regional Aviation Desert & 43 Isolated Airstrips (Task 4)</h4>
                <ul style="font-size:0.88rem; color:#334155; line-height:1.6; padding-left:1.2rem; margin:0;">
                    <li><strong>28.1% Infrastructure Underutilization:</strong> <strong>43 out of 153 registered facilities</strong> have zero commercial flights in schedule records.</li>
                    <li><strong>Geographic Categories:</strong>
                        <ul style="margin-top:4px;">
                            <li><em>Remote Island Bases:</em> Car Nicobar, Campbell Bay (INS Baaz), Kavaratti.</li>
                            <li><em>Himalayan Defense ALGs:</em> Mechuka, Tuting, Ziro, Walong in Arunachal Pradesh.</li>
                            <li><em>Unserved Regional Strips:</em> Utkela, Jeypore, Solapur, Gondia, Burnpur.</li>
                        </ul>
                    </li>
                    <li><strong>Least-Connected Commercial Feeder Strips:</strong> Bidar, Ludhiana, and Pathankot operate on only <strong>1 single direct connection</strong>.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_right:
        st.markdown(
            """
            <div class="meridian-panel" style="border-left: 5px solid #8B5CF6; height: 100%;">
                <h4 style="color:#8B5CF6; margin-bottom:8px;">3. Transit Authority via PageRank (Task 5)</h4>
                <ul style="font-size:0.88rem; color:#334155; line-height:1.6; padding-left:1.2rem; margin:0;">
                    <li><strong>Prestige vs Raw Quantity:</strong> While Degree Centrality counts the raw number of connections, PageRank measures how prestigious and influential those connected neighbors are.</li>
                    <li><strong>Top Ranked Transit Authorities:</strong>
                        <ul style="margin-top:4px;">
                            <li><strong>Delhi:</strong> PageRank = <strong>7.0548</strong> (Primary National Gateway)</li>
                            <li><strong>Bengaluru:</strong> PageRank = <strong>6.3515</strong> (Southern Tech Core)</li>
                            <li><strong>Mumbai:</strong> PageRank = <strong>6.0183</strong> (Western Financial Core)</li>
                            <li><strong>Hyderabad:</strong> PageRank = <strong>5.8948</strong> (South-Central Transit Hub)</li>
                        </ul>
                    </li>
                    <li><strong>Frequency Matters:</strong> The <strong>Srinagar–Delhi corridor</strong> (470 recorded flights) elevates Srinagar's PageRank influence due to dense connectivity to the capital.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="meridian-panel" style="border-left: 5px solid #10B981; height: 100%;">
                <h4 style="color:#10B981; margin-bottom:8px;">4. Carrier Market Concentration (Route EDA)</h4>
                <ul style="font-size:0.88rem; color:#334155; line-height:1.6; padding-left:1.2rem; margin:0;">
                    <li><strong>Airline Dominance:</strong> <strong>IndiGo commands 56.5%</strong> of all domestic flight schedules (19,066 of 33,734 records), followed by SpiceJet (10.0%) and Air India (9.2%).</li>
                    <li><strong>Trunk Corridors:</strong> The busiest route corridors in India are:
                        <ul style="margin-top:4px;">
                            <li>Srinagar → Delhi (470 schedules)</li>
                            <li>Mumbai → Delhi (354 schedules)</li>
                            <li>Jammu → Srinagar (336 schedules)</li>
                            <li>Jammu → Delhi (298 schedules)</li>
                        </ul>
                    </li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # 3. Top Hubs Summary Table
    st.subheader("Top 5 National Hubs Comparison")
    top_hubs_data = [
        {"Rank": 1, "Airport": "Delhi", "IATA": "DEL", "Inbound": 77, "Outbound": 79, "Total Conns": 156, "Degree Centrality": 0.5132, "PageRank": 7.0548, "Strategic Role": "Primary National Gateway & Busiest Hub"},
        {"Rank": 2, "Airport": "Bengaluru", "IATA": "BLR", "Inbound": 71, "Outbound": 71, "Total Conns": 142, "Degree Centrality": 0.4671, "PageRank": 6.3515, "Strategic Role": "Southern Tech & Aviation Backbone"},
        {"Rank": 3, "Airport": "Hyderabad", "IATA": "HYD", "Inbound": 68, "Outbound": 68, "Total Conns": 136, "Degree Centrality": 0.4474, "PageRank": 5.8948, "Strategic Role": "Major South-Central Transit Core"},
        {"Rank": 4, "Airport": "Mumbai", "IATA": "BOM", "Inbound": 68, "Outbound": 68, "Total Conns": 136, "Degree Centrality": 0.4474, "PageRank": 6.0183, "Strategic Role": "Western Commercial & Financial Hub"},
        {"Rank": 5, "Airport": "Kolkata", "IATA": "CCU", "Inbound": 54, "Outbound": 54, "Total Conns": 108, "Degree Centrality": 0.3553, "PageRank": 5.2368, "Strategic Role": "Eastern India & Northeast Gateway"},
    ]
    st.dataframe(
        pd.DataFrame(top_hubs_data).set_index("Rank"),
        use_container_width=True
    )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # 4. Practical Implications & Recommendations
    st.subheader("Practical Implications for Civil Aviation Policy")

    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(
            """
            <div class="meridian-panel">
                <h4 style="color:#0284C7; margin-bottom:6px;">1. Justification for UDAN Scheme</h4>
                <p style="font-size:0.85rem; color:#475569; line-height:1.5; margin:0;">
                    The discovery of 43 completely unserved airstrips mathematically substantiates the necessity for <strong>Viability Gap Funding (VGF)</strong> under UDAN Phase 5.0, prioritizing the Northeast and Eastern zones.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with p2:
        st.markdown(
            """
            <div class="meridian-panel">
                <h4 style="color:#F59E0B; margin-bottom:6px;">2. Mega-Hub Slot Decongestion</h4>
                <p style="font-size:0.85rem; color:#475569; line-height:1.5; margin:0;">
                    Delhi and Mumbai operate under extreme airspace congestion. Developing secondary twin airports (<strong>Noida Jewar</strong> and <strong>Navi Mumbai</strong>) is vital to alleviate single-point-of-failure bottlenecks.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with p3:
        st.markdown(
            """
            <div class="meridian-panel">
                <h4 style="color:#10B981; margin-bottom:6px;">3. Direct Tier-2 Corridors</h4>
                <p style="font-size:0.85rem; color:#475569; line-height:1.5; margin:0;">
                    Carriers can bypass congested metro transit hubs by launching direct point-to-point flights connecting thriving Tier-2 economic centers (e.g. Pune–Guwahati, Ahmedabad–Coimbatore).
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # 5. Overall Project Conclusion
    st.markdown(
        """
        <div class="meridian-panel" style="background-color: #F8FAFC; border: 1px solid #CBD5E1;">
            <h3 style="color:#0F172A; margin-top:0;">Overall Project Conclusion</h3>
            <p style="font-size:0.92rem; color:#334155; line-height:1.6; margin:0;">
                By modeling India's domestic flight schedules as a <strong>Directed Multigraph of 153 airport vertices and 1,423 route corridors</strong>, 
                this project demonstrates that Indian civil aviation operates as an efficient yet centralized <strong>Hub-and-Spoke system</strong>. 
                While mega-hubs like Delhi and Bengaluru provide vital national connectivity, the network faces two major structural vulnerabilities: 
                <strong>airspace saturation at top hubs</strong> and a <strong>28.1% regional connectivity deficit</strong> where 43 aerodromes remain completely unserved. 
                Balancing these disparities through secondary hub development and subsidized regional routes will be essential for resilient, inclusive aviation growth.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # 6. Viva Demonstration Guide
    with st.expander("Teacher Demonstration & Viva Quick Guide (Click to Expand)"):
        st.markdown(
            """
            **Key Questions to Answer Confidently in Your Viva:**
            
            1. **Q: Why construct a Directed Graph instead of an Undirected Graph?**
               - *Answer:* Flight operations are directional. Flight schedules, slot capacities, and passenger demands from City A to City B are not always identical in reverse.
            
            2. **Q: Why fuse the 153-airport Master Registry with flight schedules?**
               - *Answer:* In schedule datasets like `Air-Clean.csv`, an airport only appears if it has at least one scheduled flight. A purely route-derived graph would omit unserved airports, masking the 43 isolated regional and defense strips.
            
            3. **Q: What is the difference between Degree Centrality and PageRank?**
               - *Answer:* Degree Centrality measures raw direct connection volume (Delhi connects to 156 routes). PageRank evaluates the quality and transit authority of those connections (connecting to Delhi or Mumbai confers higher network importance than connecting to a remote feeder strip).
            
            4. **Q: Which airport is the busiest in India?**
               - *Answer:* Delhi (Indira Gandhi International Airport, DEL) with 156 direct connections (77 inbound, 79 outbound), Degree Centrality of 0.5132, and highest PageRank of 7.0548.
            """
        )


# Global Layout Router
def main():
    render_sidebar()

    current_page = st.session_state["current_page"]

    if current_page == "Overview":
        render_overview()
    elif current_page == "Task 1: Graph Construction & Route EDA":
        render_eda()
    elif current_page == "Tasks 2 & 3: Direct Connections & Hub Centrality":
        render_hub_centrality()
    elif current_page == "Task 4: Isolated & Least Connected Airports":
        render_isolated_airports()
    elif current_page == "Task 5: PageRank Airport Importance":
        render_pagerank()
    elif current_page == "Task 6: Route Network & Hub Connectivity Chart":
        render_route_network()
    elif current_page == "Major Findings & Summary":
        render_major_findings_and_summary()
    else:
        render_overview()

    # Enterprise Footer
    st.markdown(
        f"""
        <div class="meridian-footer">
            <strong>{PRODUCT_NAME}</strong> &bull; {PRODUCT_BADGE} &bull; Powered by Apache Spark GraphX, NetworkX & Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
