import pytest
from unittest.mock import Mock, patch, MagicMock
import os
from langchain_core.documents import Document
from rag import (
    download_video,
    transcribe_video,
    download_captions,
    split_transcript,
    retrieve,
    generate_answer,
    ingest_video,
    answer_question
)

# Fixtures
@pytest.fixture
def mock_youtube():
    with patch('rag.YouTube') as mock:
        youtube_instance = Mock()
        youtube_instance.streams.filter.return_value.first.return_value = Mock()
        
        # Create a proper captions object with get_by_language_code method
        captions_obj = Mock()
        en_caption = Mock()
        en_caption.generate_srt_captions.return_value = "1\n00:00:00,000 --> 00:00:05,000\nHello world"
        
        # Set up the get_by_language_code method
        def get_by_language_code(code):
            if code == 'en':
                return en_caption
            elif code == 'a.en':
                return None
            elif code == 'a.id':
                return None
            return None
        
        captions_obj.get_by_language_code = get_by_language_code
        youtube_instance.captions = captions_obj
        
        youtube_instance.author = "Test Author"
        youtube_instance.title = "Test Title"
        youtube_instance.description = "Test Description"
        mock.return_value = youtube_instance
        yield mock

@pytest.fixture
def mock_whisper():
    with patch('rag.whisper.load_model') as mock:
        mock_model = Mock()
        mock_model.transcribe.return_value = {"text": "Test transcription"}
        mock.return_value = mock_model
        yield mock

@pytest.fixture
def mock_vector_store():
    with patch('rag.vector_store') as mock:
        mock.similarity_search.return_value = [
            Document(page_content="Test content 1", metadata={"video_id": "test_id"}),
            Document(page_content="Test content 2", metadata={"video_id": "test_id"})
        ]
        yield mock

@pytest.fixture
def mock_llm():
    with patch('rag.llm') as mock:
        mock.invoke.return_value = Mock(content="Test answer")
        yield mock

# Test cases
def test_download_video(mock_youtube, tmp_path):
    with patch('os.path.exists') as mock_exists, \
         patch('os.makedirs') as mock_makedirs:
        mock_exists.return_value = False
        download_video("https://youtube.com/test", "test_id")
        mock_makedirs.assert_called_once_with("audio")
        mock_youtube.return_value.streams.filter.assert_called_once_with(only_audio=True)

def test_transcribe_video(mock_whisper, tmp_path):
    with patch('os.path.exists') as mock_exists, \
         patch('os.makedirs') as mock_makedirs, \
         patch('builtins.open', create=True), \
         patch('os.remove') as mock_remove:
        mock_exists.return_value = False
        transcribe_video("test_id")
        mock_makedirs.assert_called_once_with("transcripts")
        mock_whisper.assert_called_once_with("tiny", device="cpu")
        mock_remove.assert_called_once()

def test_download_captions_with_captions(mock_youtube, tmp_path):
    with patch('os.path.exists') as mock_exists, \
         patch('os.makedirs') as mock_makedirs, \
         patch('builtins.open', create=True):
        mock_exists.return_value = False
        download_captions("https://youtube.com/test", "test_id")
        mock_makedirs.assert_called_once_with("transcripts")

def test_download_captions_without_captions(mock_youtube, tmp_path):
    # Create a captions object that returns None for all language codes
    captions_obj = Mock()
    captions_obj.get_by_language_code.return_value = None
    mock_youtube.return_value.captions = captions_obj
    
    with patch('rag.download_video') as mock_download, \
         patch('rag.transcribe_video') as mock_transcribe:
        download_captions("https://youtube.com/test", "test_id")
        mock_download.assert_called_once_with("https://youtube.com/test", "test_id")
        mock_transcribe.assert_called_once_with("test_id")

def test_split_transcript(mock_youtube, tmp_path):
    with patch('langchain.document_loaders.TextLoader') as mock_loader, \
         patch('builtins.open', create=True) as mock_open:
        # Create a mock file object
        mock_file = Mock()
        mock_file.read.return_value = "Test content for transcript"
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Create a mock document
        mock_doc = Document(page_content="Test content for transcript")
        mock_loader.return_value.load.return_value = [mock_doc]
        
        result = split_transcript("test_id")
        assert len(result) > 0
        assert all(isinstance(doc, Document) for doc in result)
        assert all("video_id" in doc.metadata for doc in result)

def test_retrieve(mock_vector_store):
    result = retrieve(mock_vector_store, "test question", "test_id")
    assert len(result) == 2
    mock_vector_store.similarity_search.assert_called_once()

def test_generate_answer(mock_llm):
    docs = [
        Document(page_content="Test content 1"),
        Document(page_content="Test content 2")
    ]
    result = generate_answer("test question", docs)
    assert result == "Test answer"
    mock_llm.invoke.assert_called_once()

def test_ingest_video(mock_youtube, mock_vector_store):
    with patch('rag.download_captions') as mock_download, \
         patch('rag.split_transcript') as mock_split:
        mock_split.return_value = [Document(page_content="Test content")]
        ingest_video("https://youtube.com/test", "test_id")
        mock_download.assert_called_once_with("https://youtube.com/test", "test_id")
        mock_vector_store.add_documents.assert_called_once()

def test_answer_question(mock_vector_store, mock_llm):
    result = answer_question("test question", "test_id")
    assert result == "Test answer"
    mock_vector_store.similarity_search.assert_called_once()
    mock_llm.invoke.assert_called_once() 