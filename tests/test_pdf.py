from unittest.mock import patch
from read_document import load_document


@patch("read_document.read_pdf")
def test_load_pdf_uses_pdf_reader(mock_read_pdf, tmp_path):
    pdf_file = tmp_path / "notes.pdf"
    pdf_file.write_bytes(b"fake pdf")
    mock_read_pdf.return_value = "Python is a programming language"

    text = load_document(pdf_file)

    assert text == "Python is a programming language"
    mock_read_pdf.assert_called_once()


def test_load_txt_file(tmp_path):
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("FastAPI builds APIs", encoding="utf-8")

    assert load_document(txt_file) == "FastAPI builds APIs"
