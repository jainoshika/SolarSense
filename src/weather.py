import requests


def get_weather(lat, lon):

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}"
        f"&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m"
    )

    response = requests.get(url)

    data = response.json()

    weather = {

        "temp":
        data["current"]["temperature_2m"],

        "humidity":
        data["current"]["relative_humidity_2m"]
    }

    return weather