import pytest
from src.rag import Retriever, answer, chunk_documents


def test_retrieves_hotel_policy():
    retriever = Retriever(chunk_documents('data'))
    hits = retriever.search('hotel reimbursement limit')
    assert hits
    assert 'sample_policy.md' == hits[0]['source']


def test_abstains_on_unknown_topic():
    retriever = Retriever(chunk_documents('data'))
    result = answer('quantum entanglement particle physics', retriever)
    assert result['mode'] == 'abstain'


def test_question_validation():
    with pytest.raises(ValueError):
        answer(' ', Retriever(chunk_documents('data')))
