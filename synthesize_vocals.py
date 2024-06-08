import os
import torch
from TTS.api import TTS

# Initialize the TTS model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=False)

# Function to synthesize vocals
def synthesize_vocals(text, output_path):
    # Generate the audio
    tts.tts_to_file(text=text, file_path=output_path)

# Example usage
lyrics = """
Goli maaaaaaar... goli maar!
Goli mar-mar-mar-mar-mar-mar-(mar)...
Goli mar-mar-mar-mar-mar-(mar).....
Kashno rau kaugileste yenche staavooo...
Ne pali manta meste yee maltovooo.....
Kandaru padda kane shungaa.. raamaaaa...
Valkunga guda yesta vaya raamaaa...
Goli maar!
Goli.. mar-mar-mar-mar-(mar)....
Rrrrrgrrryyaaaaayaaaaa!
Puttan gane!
Matte koye!
Kutte chan taan vaaste...
Muttu ta taa lantee...
Alla e potaabe anno luuu....
Balla ma ti che staaro...
Alla lika ti staaro....
Unna ma ti poya ka!
Uuhu paata restaaro..
Ouuu.. iltii baltii pubantii...
Ouuu.. shaaltii shantii womchantii..
"""

# Split the lyrics into smaller segments
lyrics_segments = [
    "Goli maaaaaaar... goli maar!",
    "Goli mar-mar...",
    "mar-mar...",
    "mar-(mar)...",
    "Goli mar-mar...",
    "mar-mar-(mar).....",
    "Kashno rau kaugileste yenche staavooo...",
    "Ne pali manta meste yee maltovooo.....",
    "Kandaru padda kane shungaa.. raamaaaa...",
    "Valkunga guda yesta vaya raamaaa...",
    "Goli maar!",
    "Goli.. mar-mar-mar-mar-(mar)....",
    "Rrrrrgrrryyaaaaayaaaaa!",
    "Puttan gane!",
    "Matte koye!",
    "Kutte chan taan vaaste...",
    "Muttu ta taa lantee...",
    "Alla e potaabe anno luuu....",
    "Balla ma ti che staaro...",
    "Alla lika ti staaro....",
    "Unna ma ti poya ka!",
    "Uuhu paata restaaro..",
    "Ouuu.. iltii baltii pubantii...",
    "Ouuu.. shaaltii shantii womchantii..",
    "Ruddem, rouddem, ririim saa..",
    "Murkam burdam mumuur-shaa...",
    "Killeeer-killeer-killeer...",
    "killeer-killeer...",
    "Goli maar!",
    "Goli maar-maar-maar...",
    "maar-maar-maar-(maar)...",
    "Killeeer-killeer-killeer...",
    "killeer-killeer...",
    "Muttan gane!",
    "Nitte koye!",
    "Korvi da yaa laaste...",
    "Kongu lan tuu kuunte..",
    "Challa re dagda go ii puuduuu..",
    "Cheta vadi chestaaro..",
    "Kordii merda kostaaro..",
    "Shma shanala viduyo..",
    "Isha chaalu partaaro...",
    "Ouu, naari, ghyaari.. vayaari..",
    "Ouu, tagga, kaali, kankaadi!",
    "Tiivruum, tiindruum dibdrug shaa..",
    "Munduu vinuka parik shaa..",
    "Killeeer-killeer-killeer...",
    "killeer-killeer...",
    "Goli maar!",
    "Goli maar-maar-maar...",
    "maar-maar-(maar)....",
    "Kashno rau kaugileste enche stavooo...",
    "Ne pali manta meste ye maltovooo.....",
    "Kandaru padda kane shungaa.. raamaaaa...",
    "Valkunga guda yesta vaya raamaaa...",
    "Gooli maar! ... maar-maar-maar...",
    "maar-maar-(maar)....",
    "Ayyyeeeeeeee!!!!!"
]

# Synthesize each segment and save to individual files
for i, segment in enumerate(lyrics_segments):
    output_path = f"golimaar_synthesized_segment_{i}.wav"
    synthesize_vocals(segment, output_path)
    print(f"Synthesized vocals saved to {output_path}")
