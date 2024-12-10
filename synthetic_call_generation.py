import os
import azure.cognitiveservices.speech as speechsdk
from pydub import AudioSegment
# load environment variables
from dotenv import load_dotenv
load_dotenv()

call_transcript = """
Bank Advisor: Good afternoon, thank you for calling ABC Bank. My name is Alex. How can I assist you today?

Customer: Hi Alex, I'm interested in learning about the bank's options for conservative investments. Can you help me with that?

Bank Advisor: Of course! I'd be happy to help. Are you looking for short-term or long-term investment options?

Customer: I'm more interested in long-term investments that have lower risk.

Bank Advisor: Great, we have several options that might suit your needs. One of our most popular conservative investment options is our fixed deposit account. It offers a guaranteed return with minimal risk.

Customer: That sounds interesting. Can you tell me more about the fixed deposit account?

Bank Advisor: Certainly. With a fixed deposit account, you can choose a term ranging from one year to five years. The interest rate is fixed for the duration of the term, and you can earn a higher return compared to a regular savings account.

Customer: What is the current interest rate for a five-year fixed deposit?

Bank Advisor: The current interest rate for a five-year fixed deposit is 3.5% per annum. This rate is guaranteed for the entire term, providing you with a stable and predictable return.

Customer: That sounds good. Are there any other conservative investment options available?

Bank Advisor: Yes, we also offer government bonds and high-quality corporate bonds. These bonds are considered low-risk and provide regular interest payments. Additionally, we have a conservative investment fund that diversifies your investment across various low-risk assets.

Customer: I'm interested in the conservative investment fund. Can you explain how it works?

Bank Advisor: Of course. The conservative investment fund is managed by our team of experts who invest in a mix of government bonds, high-quality corporate bonds, and other low-risk assets. The goal is to provide steady returns while minimizing risk. You can start with a minimum investment of $1,000.

Customer: That sounds like a good option. How can I get started with the conservative investment fund?

Bank Advisor: You can visit any of our branches to speak with an investment advisor, or you can apply online through our website. If you choose to apply online, you'll need to provide some basic information and complete a risk assessment questionnaire.

Customer: Thank you, Alex. I appreciate your help. I'll consider my options and get back to you.

Bank Advisor: You're welcome. If you have any other questions or need further assistance, please don't hesitate to contact us. Have a great day!

Customer: You too! Goodbye.
"""

advisor_utterances = [] 
customer_utterances = []
for utterance in call_transcript.split('\n'):
    if utterance.startswith('Bank Advisor:'):
        advisor_utterances.append(utterance.replace('Bank Advisor:', '').strip())
    elif utterance.startswith('Customer:'):
        customer_utterances.append(utterance.replace('Customer:', '').strip())



# This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('AZURE_SPEECH_KEY'), region=os.environ.get('AZURE_SPEECH_REGION'))
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

advisor_synthetic_voice = 'en-US-AvaMultilingualNeural'
customer_synthetic_voice = 'en-US-AndrewMultilingualNeural'

# Configure speech synthesis output format for high quality audio
speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm)

# Configure advisor voice
speech_config.speech_synthesis_voice_name = advisor_synthetic_voice
advisor_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

# Configure customer voice 
speech_config.speech_synthesis_voice_name = customer_synthetic_voice
customer_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

# Synthesize advisor utterances
advisor_segments = []
for i, text in enumerate(advisor_utterances):
    result = advisor_synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        stream = speechsdk.AudioDataStream(result)
        temp_file = f'advisor_utterance_{i}.wav'
        stream.save_to_wav_file(temp_file)
        advisor_segments.append(AudioSegment.from_wav(temp_file))
        os.remove(temp_file)  # Clean up temporary file
    else:
        print(f"Error synthesizing advisor speech: {result.cancellation_details.reason}")
        if result.cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"Error details: {result.cancellation_details.error_details}")

# Synthesize customer utterances  
customer_segments = []
for i, text in enumerate(customer_utterances):
    result = customer_synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        stream = speechsdk.AudioDataStream(result)
        temp_file = f'customer_utterance_{i}.wav'
        stream.save_to_wav_file(temp_file)
        customer_segments.append(AudioSegment.from_wav(temp_file))
        os.remove(temp_file)  # Clean up temporary file
    else:
        print(f"Error synthesizing customer speech: {result.cancellation_details.reason}")
        if result.cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"Error details: {result.cancellation_details.error_details}")

# Combine all audio segments in conversation order
combined_audio = AudioSegment.empty()
for advisor_segment, customer_segment in zip(advisor_segments, customer_segments):
    combined_audio += advisor_segment
    combined_audio += customer_segment

# Export the combined audio
combined_audio.export("synthetic_call.wav", format="wav")

