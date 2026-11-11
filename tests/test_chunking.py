from read_document import split_into_chunks
import pytest
def test_split_into_words():
    assert split_into_chunks("",5,1) == []

def test_invalid_overlap_large():
    with pytest.raises(ValueError):
        split_into_chunks("Python is used for testing", 2,3)

def test_invalid_chunk_size_zero():
    with pytest.raises(ValueError):
        split_into_chunks("Python is used for testing", 0,3)

def test_overlap_negative():
    with pytest.raises(ValueError):
        split_into_chunks("Python is used for machine learning", 2, -1)

def test_overlap_chunk_size_same():
    with pytest.raises(ValueError):
        split_into_chunks("Python is used for training Ai", 2,2)

def test_split_words_correctly():
    assert split_into_chunks("A B C D E F",2,1) == [
        ["A", "B"],
        ["B", "C"],
        ["C", "D"],
        ["D", "E"],
        ["E", "F"],
    ]

def test_basic_chunks():
    assert split_into_chunks("A B C D E F", 2, 0) == [
        ['A', 'B'],
        ['C', 'D'],
        ['E', 'F']
    ]

