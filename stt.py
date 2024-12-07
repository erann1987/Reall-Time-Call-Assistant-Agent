import os
import time
import queue
import threading
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
load_dotenv()
from streamlit.runtime.scriptrunner import add_script_run_ctx


class TranscriptionManager:
    def __init__(self):
        self.transcription_queue = queue.Queue()
        self.is_running = False
        self._consumer_callback = None
        self.complete_transcription = []

    def set_consumer_callback(self, callback):
        """Set the callback function that will process transcriptions"""
        self._consumer_callback = callback

    def get_complete_transcription(self):
        """Get the complete transcription as a list of dictionaries"""
        return self.complete_transcription

    def _default_consumer_thread(self):
        """Default consumer function that processes transcriptions from the queue"""
        while self.is_running:
            try:
                # Get transcription from queue, waiting if necessary
                transcription = self.transcription_queue.get(timeout=1)
                
                # Check for stop signal
                if transcription is None:
                    print("Consumer received stop signal")
                    break
                    
                # Store final transcriptions
                if transcription['type'] == 'final':
                    self.complete_transcription.append(transcription)
                
                # Process the transcription using callback if available
                if self._consumer_callback:
                    self._consumer_callback(transcription)
                else:
                    print(f"\nReceived {transcription['type']} transcription:")
                    print(f"\tText: {transcription['text']}")
                    print(f"\tSpeaker ID: {transcription['speaker_id']}")
                
                # Mark task as done
                self.transcription_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in consumer: {e}")

    def start_consumer(self):
        """Start the consumer thread"""
        self.is_running = True
        self.complete_transcription = []  # Reset transcription at start
        self.consumer_thread = threading.Thread(
            target=self._default_consumer_thread,
            name="TranscriptionConsumer",
            daemon=True,
        )
        add_script_run_ctx(self.consumer_thread)
        self.consumer_thread.start()

    def stop_consumer(self):
        """Stop the consumer thread"""
        self.is_running = False
        self.transcription_queue.put(None)
        if hasattr(self, 'consumer_thread'):
            self.consumer_thread.join()

# Create a global instance
transcription_manager = TranscriptionManager()

def conversation_transcriber_recognition_canceled_cb(evt: speechsdk.SessionEventArgs):
    print('Canceled event')

def conversation_transcriber_session_stopped_cb(evt: speechsdk.SessionEventArgs):
    print('SessionStopped event')
    # Add None to signal the consumer to stop
    transcription_manager.transcription_queue.put(None)

def conversation_transcriber_transcribed_cb(evt: speechsdk.SpeechRecognitionEventArgs):
    if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
        transcription = {
            'text': evt.result.text,
            'speaker_id': evt.result.speaker_id,
            'type': 'final'
        }
        transcription_manager.transcription_queue.put(transcription)

def conversation_transcriber_transcribing_cb(evt: speechsdk.SpeechRecognitionEventArgs):
    transcription = {
        'text': evt.result.text,
        'speaker_id': evt.result.speaker_id,
        'type': 'interim'
    }
    transcription_manager.transcription_queue.put(transcription)

def conversation_transcriber_session_started_cb(evt: speechsdk.SessionEventArgs):
    print('SessionStarted event')

def recognize_from_file(file_path):
    # This example requires environment variables named "AZURE_SPEECH_KEY" and "AZURE_SPEECH_REGION"
    speech_config = speechsdk.SpeechConfig(
        subscription=os.environ.get('AZURE_SPEECH_KEY'), 
        region=os.environ.get('AZURE_SPEECH_REGION')
    )
    speech_config.speech_recognition_language="en-US"
    speech_config.set_property(
        property_id=speechsdk.PropertyId.SpeechServiceResponse_DiarizeIntermediateResults,
        value='true'
    )

    audio_config = speechsdk.audio.AudioConfig(filename=file_path)
    conversation_transcriber = speechsdk.transcription.ConversationTranscriber(
        speech_config=speech_config,
        audio_config=audio_config
    )

    transcribing_stop = False

    def stop_cb(evt: speechsdk.SessionEventArgs):
        print('CLOSING on {}'.format(evt))
        nonlocal transcribing_stop
        transcribing_stop = True

    # Connect callbacks to the events
    conversation_transcriber.transcribed.connect(conversation_transcriber_transcribed_cb)
    conversation_transcriber.transcribing.connect(conversation_transcriber_transcribing_cb)
    conversation_transcriber.session_started.connect(conversation_transcriber_session_started_cb)
    conversation_transcriber.session_stopped.connect(conversation_transcriber_session_stopped_cb)
    conversation_transcriber.canceled.connect(conversation_transcriber_recognition_canceled_cb)
    conversation_transcriber.session_stopped.connect(stop_cb)
    conversation_transcriber.canceled.connect(stop_cb)

    # Start the consumer thread
    transcription_manager.start_consumer()

    # Start transcribing
    conversation_transcriber.start_transcribing_async()

    # Wait for completion
    while not transcribing_stop:
        time.sleep(.5)

    conversation_transcriber.stop_transcribing_async()
    
    # Stop and wait for the consumer thread to finish
    transcription_manager.stop_consumer()

# Main execution
if __name__ == "__main__":
    try:
        file_path = "audio.wav"
        recognize_from_file(file_path)
    except Exception as err:
        print("Encountered exception. {}".format(err))

