from errors import classify_error, ErrorClass
from parsing import ParseError

def test_timeout_is_retryable():
    assert classify_error(TimeoutError("vllm timeout")) == ErrorClass.RETRYABLE

def test_connection_is_retryable():
    assert classify_error(ConnectionError("conn refused")) == ErrorClass.RETRYABLE

def test_http_5xx_is_retryable():
    class E(Exception):
        status_code = 503
    assert classify_error(E("server error")) == ErrorClass.RETRYABLE

def test_http_400_is_terminal():
    class E(Exception):
        status_code = 400
    assert classify_error(E("bad request / context overflow")) == ErrorClass.TERMINAL

def test_parse_error_is_terminal():
    assert classify_error(ParseError("irreparable")) == ErrorClass.TERMINAL
