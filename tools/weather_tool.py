from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool
import requests
from datetime import datetime

class WeatherInput(BaseModel):
    location: str = Field(description="The location to get the weather for")
    specific_info: str = Field(default=None, description="Specific information to filter (e.g., 'humidity', 'temperature', etc.)")

def get_weather(location: str, specific_info: str = None) -> str:
    api_key = 'f3df87956e90eae64c886028a3a5b715'
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        weather_info = {
            "description": data['weather'][0]['description'].capitalize(),
            "temperature": f"{data['main']['temp']}°C",
            "feels like": f"{data['main']['feels_like']}°C",
            "humidity": f"{data['main']['humidity']}%",
            "wind speed": f"{data['wind']['speed']} m/s",
            "sunrise": datetime.utcfromtimestamp(data['sys']['sunrise']).strftime('%Y-%m-%d %H:%M:%S UTC'),
            "sunset": datetime.utcfromtimestamp(data['sys']['sunset']).strftime('%Y-%m-%d %H:%M:%S UTC')
        }
        
        if specific_info:
            specific_info = specific_info.lower()
            if specific_info in weather_info:
                return f"The {specific_info} in {location} is {weather_info[specific_info]}."
            else:
                return f"Sorry, I could not find the {specific_info} information for {location}."

        weather_report = (
            f"Weather in {location}:\n"
            f"- Description: {weather_info['description']}\n"
            f"- Temperature: {weather_info['temperature']} (feels like {weather_info['feels like']})\n"
            f"- Humidity: {weather_info['humidity']}\n"
            f"- Wind Speed: {weather_info['wind speed']}\n"
            f"- Sunrise: {weather_info['sunrise']}\n"
            f"- Sunset: {weather_info['sunset']}"
        )
        return weather_report
    else:
        return f"Could not retrieve weather data for {location}"
    
def get_weather_tool(location: str, specific_info: str = None) -> str:
    return get_weather(location, specific_info)

weather_tool = StructuredTool.from_function(
    func=get_weather_tool,
    name="weather_tool",
    description="Get the current weather for a specific location, with an option to filter specific information like humidity, temperature, wind speed, etc.",
    args_schema=WeatherInput,
    return_direct=True,
)
