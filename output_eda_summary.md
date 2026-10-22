# Indian Airline Route Network & Master Airports - EDA Summary

## 1. Flight Schedule Dataset (`data/Air-Clean.csv`)
- **Total Records:** 33,734 schedule records (2018–2025)
- **Unique Directed Flight Routes:** 1,423 corridors
- **Airlines Represented:** 14 (IndiGo, Air India, SpiceJet, GoAir, Vistara, etc.)
- **Active Airport Vertices:** 110 cities

### Top 10 Flight Corridors by Frequency
| Origin | Destination | Recorded Schedules |
| :--- | :--- | :--- |
| Srinagar | Delhi | 470 |
| Mumbai | Delhi | 354 |
| Jammu | Srinagar | 336 |
| Jammu | Delhi | 298 |
| Delhi | Mumbai | 285 |
| Bengaluru | Delhi | 283 |
| Srinagar | Jammu | 274 |
| Mumbai | Bengaluru | 248 |
| Bengaluru | Mumbai | 217 |
| Hyderabad | Bengaluru | 212 |

## 2. Airport Master Registry (`data/indian_airports_master.csv`)
- **Total Registered Vertices:** 153
- **Commercial Vertices:** 110
- **Genuinely Isolated Vertices (0 Flights):** 43

### Aviation Significance of Isolated Vertices
In real-world airline networks, calculating graph metrics solely from flight schedule tables (`Air-Clean.csv`)
introduces selection bias: an unserved airport never appears as an origin or destination.
By compiling and fusing the **Master Airport Registry** (153 facilities) as graph vertices, GraphX and NetworkX
accurately measure both high-centrality mega-hubs (Delhi, Mumbai, Bengaluru) and zero-degree isolated regional strips
(Car Nicobar, Utkela, Mechuka, Solapur, Gondia).
