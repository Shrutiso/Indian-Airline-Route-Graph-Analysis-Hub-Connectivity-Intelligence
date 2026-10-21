# Indian Airline Route Graph & Airport Master Dataset - EDA Summary

## 1. Flight Schedule Dataset (`data/Air-Clean.csv`)
- **Total Records:** 33,734 schedule records (2018–2025)
- **Unique Directed Flight Routes:** 1,423 routes
- **Airlines Represented:** 14 (IndiGo, Air India, SpiceJet, GoAir, Vistara, AirAsia, etc.)
- **Active Airport Vertices:** 110 locations

### Top Flight Corridors by Frequency
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
- **Commercial Vertices:** 0
- **Isolated Vertices:** 43

### Why Isolated Vertices Matter in Graph Analytics
In real-world airline and transit systems, relying only on schedule edge tables (`Air-Clean.csv`) conceals non-operational airports, UDAN regional strips, defense airbases, and unserved locations. By fusing the **Master Airport Registry** as vertices, GraphX correctly evaluates both high-centrality mega-hubs (Delhi, Mumbai, Bengaluru) and zero-degree isolated nodes (Car Nicobar, Gondia, Utkela, Mechuka, Solapur).
