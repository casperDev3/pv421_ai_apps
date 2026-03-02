from google import genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

response = client.models.generate_content(
    model="gemini-3-flash-preview", contents="Привіт! Як справи?"
)
print(response.text)

from google import genai
from google.genai import types
import wave

# Set up the wave file to save the output:
def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
   with wave.open(filename, "wb") as wf:
      wf.setnchannels(channels)
      wf.setsampwidth(sample_width)
      wf.setframerate(rate)
      wf.writeframes(pcm)

client = genai.Client()
#
# response = client.models.generate_content(
#    model="gemini-2.5-flash-preview-tts",
#    contents="Скажи сумно: Який чарівний суботній день!",
#    config=types.GenerateContentConfig(
#       response_modalities=["AUDIO"],
#       speech_config=types.SpeechConfig(
#          voice_config=types.VoiceConfig(
#             prebuilt_voice_config=types.PrebuiltVoiceConfig(
#                voice_name='Kore',
#             )
#          )
#       ),
#    )
# )
#
# data = response.candidates[0].content.parts[0].inline_data.data
#
# file_name='out.wav'
# wave_file(file_name, data) # Saves the file to current directory

# import asyncio
# from google import genai
# from google.genai import types
#
# client = genai.Client(http_options={'api_version': 'v1alpha'})
#
#
# async def main():
#     async def receive_audio(session):
#         """Example background task to process incoming audio."""
#         while True:
#             async for message in session.receive():
#                 audio_data = message.server_content.audio_chunks[0].data
#                 # Process audio...
#                 await asyncio.sleep(10 ** -12)
#
#     async with (
#         client.aio.live.music.connect(model='models/lyria-realtime-exp') as session,
#         asyncio.TaskGroup() as tg,
#     ):
#         # Set up task to receive server messages.
#         tg.create_task(receive_audio(session))
#
#         # Send initial prompts and config
#         await session.set_weighted_prompts(
#             prompts=[
#                 types.WeightedPrompt(text='minimal techno', weight=1.0),
#             ]
#         )
#         await session.set_music_generation_config(
#             config=types.LiveMusicGenerationConfig(bpm=90, temperature=1.0)
#         )
#
#         # Start streaming music
#         await session.play()
#
#
# if __name__ == "__main__":
#     asyncio.run(main())

# import time
# from google import genai
#
# client = genai.Client()
# prompt = "A whimsical stop-motion animation of a tiny robot tending to a garden of glowing mushrooms on a miniature planet."
#
# operation = client.models.generate_videos(
#     model="veo-3.1-generate-preview",
#     prompt=prompt,
# )
#
# # Poll the operation status until the video is ready.
# while not operation.done:
#     print("Waiting for video generation to complete...")
#     time.sleep(10)
#     operation = client.operations.get(operation)
#
# # Download the generated video.
# generated_video = operation.response.generated_videos[0]
# client.files.download(file=generated_video.video)
# generated_video.video.save("style_example.mp4")
# print("Generated video saved to style_example.mp4")