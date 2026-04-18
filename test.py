from openai import OpenAI

client = OpenAI(base_url="http://localhost:5000/v1", api_key="any")

response = client.audio.speech.create(
       model="gpt-sovits",
       voice="default",
       input="man. what can i say. mamba out",
       extra_body={"lang": "en", "speed" :"1"}
     )
response.stream_to_file("speech.mp3")