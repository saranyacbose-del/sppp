# Cached API Responses

This directory contains cached API responses used in Phase 2 validation.
Data collected over five weekday mornings, 08:00-10:00 IST.

Files:
  osm_graph.graphml          - 847-node OSM road graph (osmnx)
  mapbox_traffic.json        - Mapbox Traffic API v2 congestion -> O1 (Table 5)
  openweathermap.json        - OpenWeatherMap precipitation -> O3 (Table 6)
  obstacle_scores_osm.json   - Per-node O1-O8 severity scores for Phase 2

To regenerate: set MAPBOX_TOKEN and OWM_API_KEY env vars then run
  python experiments/run_phase2_osm.py
