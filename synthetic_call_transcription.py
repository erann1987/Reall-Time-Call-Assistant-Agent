import yaml
from pydantic import BaseModel, Field
from typing import Literal
import dspy
import os
import azure.cognitiveservices.speech as speechsdk
from pydub import AudioSegment
import argparse
from dotenv import load_dotenv
load_dotenv()


class CallTranscription(BaseModel):
    speaker: Literal["advisor", "customer"] = Field(description="The speaker of the utterance")
    utterance: str
    relevant_notes: list[str] = Field(
        description="""Optional relevant notes corresponding to specific utterances (if applicable). 
        Each note should:
        1. Be written in present tense, as though it was documented immediately after the previous meeting ended
        2. Represent detailed free text written by the advisor after previous calls with the client
        3. Include key points discussed in those prior interactions
        4. Reflect the client's intent during those calls
        5. Outline any actions or steps taken by the advisor based on those discussions
        Each relevant note must be directly related to the corresponding utterance, providing additional context or background that aligns with the current call's conversation.""",
        default=[]
    )

class SyntheticCallTranscription(dspy.Signature):
    """Create a synthetic call transcription that simulates a realistic conversation between a bank advisor and a customer, centered around a specified topic. The transcription should include the following elements:
	1.	Greeting and Introduction: One participant greets the other and introduces themselves.
	2.	Purpose of the Call: The other participant explains their reason for the call or inquiry.
	3.	Detailed Discussion: A back-and-forth conversation on the specified topic, with the advisor providing information, advice, or support, and the customer asking questions or seeking clarification.
	4.	Follow-Up Questions: The customer asks additional questions, and the advisor responds with detailed explanations or solutions.
	5.	Relevant Notes Integration: Some utterances in the conversation should connect to notes that the advisor has written after previous interactions with the customer. These notes should be realistic, detailed, and relevant to the current discussion.
	6.	Summary and Conclusion: The advisor summarizes the key points discussed during the call.
	7.	Closing Statements: Both participants exchange farewell statements.
    The transcription should be natural in flow, length, and content, reflecting a typical customer service interaction.
    Example:
    Advisor: Good afternoon, thank you for calling ABC Bank. My name is Alex. How can I assist you today?
    Customer: Hi Alex, I'm interested in learning about the bank's options for conservative investments. Can you help me with that?
    Advisor: Of course! I'd be happy to help. Are you looking for short-term or long-term investment options?
    Customer: I'm more interested in long-term investments that have lower risk.
    Advisor: Great, we have several options that might suit your needs. One of our most popular conservative investment options is our fixed deposit account. It offers a guaranteed return with minimal risk.
    Customer: That sounds interesting. Can you tell me more about the fixed deposit account?
    Advisor: Certainly. With a fixed deposit account, you can choose a term ranging from one year to five years. The interest rate is fixed for the duration of the term, and you can earn a higher return compared to a regular savings account.
    Customer: What is the current interest rate for a five-year fixed deposit?
    Advisor: The current interest rate for a five-year fixed deposit is 3.5% per annum. This rate is guaranteed for the entire term, providing you with a stable and predictable return.
    Customer: That sounds good. Are there any other conservative investment options available?
    Advisor: Yes, we also offer government bonds and high-quality corporate bonds. These bonds are considered low-risk and provide regular interest payments. Additionally, we have a conservative investment fund that diversifies your investment across various low-risk assets.
    Customer: I'm interested in the conservative investment fund. Can you explain how it works?
    Advisor: Of course. The conservative investment fund is managed by our team of experts who invest in a mix of government bonds, high-quality corporate bonds, and other low-risk assets. The goal is to provide steady returns while minimizing risk. You can start with a minimum investment of $1,000.
    Customer: That sounds like a good option. How can I get started with the conservative investment fund?
    Advisor: You can visit any of our branches to speak with an investment advisor, or you can apply online through our website. If you choose to apply online, you'll need to provide some basic information and complete a risk assessment questionnaire.
    Customer: Thank you, Alex. I appreciate your help. I'll consider my options and get back to you.
    Advisor: You're welcome. If you have any other questions or need further assistance, please don't hesitate to contact us. Have a great day!
    Customer: You too! Goodbye."""
    topic: str = dspy.InputField(desc="The topic of the call")
    call_transcriptions: list[CallTranscription] = dspy.OutputField(desc="A list of call transcriptions between a client and a bank advisor based on the topic")
    # non_relevant_notes: list[str] = dspy.OutputField(
    #     desc="""A list of non-relevant notes to accompany the following call transcription. 
    #     These notes should:
    #     1. Be written in present tense, as if actively documenting the advisor's current understanding of the client's situation
    #     2. Be detailed and include key points discussed during prior calls
    #     3. Highlight the client's intent during those past interactions
    #     4. Describe any actions or steps taken by the advisor in response to those earlier calls
    #     The notes must pertain to previous calls on entirely different topics, ensuring they are unrelated to the current call transcription""",
    # )


def generate_audio_file(call_transcriptions, data_path):
    advisor_utterances = [transcription.utterance for transcription in call_transcriptions if transcription.speaker == "advisor"]
    customer_utterances = [transcription.utterance for transcription in call_transcriptions if transcription.speaker == "customer"]

    speech_config = speechsdk.SpeechConfig(
        subscription=os.environ.get('AZURE_SPEECH_KEY'),
        region=os.environ.get('AZURE_SPEECH_REGION')
    )

    advisor_synthetic_voice = 'en-US-AvaMultilingualNeural'
    customer_synthetic_voice = 'en-US-AndrewMultilingualNeural'

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
    combined_audio.export(f"{data_path}/call_recording.wav", format="wav")


class NoneRelevantNotesGenerator(dspy.Signature):
    """Generate a list of non-relevant notes to accompany the following call transcription. 
        These notes should:
        1. Be written in present tense, as if actively documenting the advisor's current understanding of the client's situation
        2. Be detailed and include key points discussed during prior calls
        3. Highlight the client's intent during those past interactions
        4. Describe any actions or steps taken by the advisor in response to those earlier calls
        The notes must pertain to previous calls on entirely different topics, ensuring they are unrelated to the current call transcription"""
    topic: str = dspy.InputField()
    call_transcriptions: list[CallTranscription] = dspy.InputField()
    non_relevant_notes: list[str] = dspy.OutputField(desc="List of notes from previous calls unrelated to the current topic")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic call data')
    parser.add_argument('--conversation-topic', type=str, required=True,
                      help='The topic of the conversation')
    args = parser.parse_args()

    # Load config
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    lm = dspy.LM(
        model=f"azure/{config.get('azure_deployment_model')}",
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        api_base=os.getenv('AZURE_OPENAI_API_BASE'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
        cache=False
    )
    dspy.configure(lm=lm)

    call_topic = args.conversation_topic

    transcription_generator = dspy.ChainOfThought(SyntheticCallTranscription)
    result = transcription_generator(topic=call_topic, temperature=0.9)

    none_relevant_notes_generator = dspy.ChainOfThought(NoneRelevantNotesGenerator)
    non_relevant_notes = none_relevant_notes_generator(
        topic=call_topic, 
        call_transcriptions=result.call_transcriptions,
        temperature=0.9
    ).non_relevant_notes

    # get relevant notes from the call transcriptions and concatenate them
    relevant_notes = [note for transcription in result.call_transcriptions for note in transcription.relevant_notes]
    total_notes = relevant_notes + non_relevant_notes

    # create a folder for the topic if it doesn't exist
    data_path = f"synthetic_data/{call_topic}"
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # save transcriptions to jsonl
    with open(f'{data_path}/call_transcriptions.jsonl', 'w') as file:
        for transcription in result.call_transcriptions:
            file.write(transcription.model_dump_json() + '\n')

    with open(f'{data_path}/call_notes.txt', 'w') as file:
        for note in total_notes:
            file.write(note + '\n')

    generate_audio_file(result.call_transcriptions, data_path)

if __name__ == "__main__":
    main()