import flightradar24
import json
fr = flightradar24.Api()
flight_id = "DY7202"
print(json.dumps(fr.get_flight(flight_id), indent=4))