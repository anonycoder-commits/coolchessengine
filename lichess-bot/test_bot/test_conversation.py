"""Tests for the PERSONALITY FEATURE: the one-time material-lead taunt in lib/conversation.py."""

import chess
import chess.engine
from lib.conversation import Conversation, is_winning_big, MIN_MOVE_NUMBER_FOR_TAUNT, TAUNT_SCORE_THRESHOLD_CP


class FakeEngine:
    """Minimal stand-in for EngineWrapper: is_winning_big only ever reads .scores."""

    def __init__(self, scores: list[chess.engine.PovScore]) -> None:
        """Store the fake score history."""
        self.scores = scores


class FakeGame:
    """Minimal stand-in for model.Game: send_reply reads .url(), .id, and .username."""

    id = "testgame"
    username = "coolchess3000"

    def url(self) -> str:
        """Return a fake game URL, matching model.Game's interface."""
        return "https://lichess.org/testgame"


class FakeLichess:
    """Records every chat call instead of hitting the network."""

    def __init__(self) -> None:
        """Start with an empty call log."""
        self.chat_calls: list[tuple[str, str, str]] = []

    def chat(self, game_id: str, room: str, text: str) -> None:
        """Record the call instead of sending it."""
        self.chat_calls.append((game_id, room, text))


def board_at_move(move_number: int) -> chess.Board:
    """Build a legal board sitting at approximately the given fullmove number."""
    board = chess.Board()
    # Alternate a harmless repeating knight shuffle so the board stays legal and reaches
    # the target fullmove_number without needing a real game script.
    moves = ["Nf3", "Nf6", "Ng1", "Ng8"]
    i = 0
    while board.fullmove_number < move_number:
        board.push_san(moves[i % len(moves)])
        i += 1
    return board


# --- is_winning_big ---

def test_is_winning_big_false_with_no_scores() -> None:
    """No search has happened yet -> never a taunt."""
    engine = FakeEngine(scores=[])
    board = board_at_move(MIN_MOVE_NUMBER_FOR_TAUNT + 1)
    assert is_winning_big(engine, board) is False


def test_is_winning_big_false_before_the_move_gate_even_with_a_huge_score() -> None:
    """A huge score in the opening shouldn't fire -- avoids early eval/book noise."""
    board = chess.Board()  # fullmove_number == 1
    engine = FakeEngine(scores=[chess.engine.PovScore(chess.engine.Cp(2000), chess.WHITE)])
    assert is_winning_big(engine, board) is False


def test_is_winning_big_true_above_threshold_past_the_move_gate() -> None:
    """Past the move gate, a score clearly above the threshold should fire."""
    board = board_at_move(MIN_MOVE_NUMBER_FOR_TAUNT + 1)
    engine = FakeEngine(scores=[chess.engine.PovScore(chess.engine.Cp(TAUNT_SCORE_THRESHOLD_CP + 50), chess.WHITE)])
    assert is_winning_big(engine, board) is True


def test_is_winning_big_false_below_threshold_past_the_move_gate() -> None:
    """Past the move gate, a modest score should NOT fire."""
    board = board_at_move(MIN_MOVE_NUMBER_FOR_TAUNT + 1)
    engine = FakeEngine(scores=[chess.engine.PovScore(chess.engine.Cp(TAUNT_SCORE_THRESHOLD_CP - 400), chess.WHITE)])
    assert is_winning_big(engine, board) is False


def test_is_winning_big_true_for_a_mate_score() -> None:
    """A found mate for us must count as a big lead -- exercises the mate_score fallback
    (bare .score() returns None for a mate score; the code must never call it bare)."""
    board = board_at_move(MIN_MOVE_NUMBER_FOR_TAUNT + 1)
    engine = FakeEngine(scores=[chess.engine.PovScore(chess.engine.Mate(4), chess.WHITE)])
    assert is_winning_big(engine, board) is True


# --- Conversation.maybe_send_taunt ---

def test_maybe_send_taunt_fires_once_and_then_stays_silent() -> None:
    """The guard must allow exactly one taunt per Conversation instance."""
    board = board_at_move(MIN_MOVE_NUMBER_FOR_TAUNT + 1)
    engine = FakeEngine(scores=[chess.engine.PovScore(chess.engine.Cp(TAUNT_SCORE_THRESHOLD_CP + 100), chess.WHITE)])
    li = FakeLichess()
    conversation = Conversation(FakeGame(), engine, li, "test-version", [])  # type: ignore[arg-type]

    assert conversation.has_taunted is False
    assert len(li.chat_calls) == 0

    conversation.maybe_send_taunt(board)
    assert conversation.has_taunted is True
    assert len(li.chat_calls) == 1
    game_id, room, text = li.chat_calls[0]
    assert game_id == "testgame"
    assert room == "player"
    assert text  # a real taunt string was chosen, not empty

    # Still winning big on a later call -- must NOT send a second message.
    conversation.maybe_send_taunt(board)
    assert len(li.chat_calls) == 1


def test_maybe_send_taunt_does_nothing_when_not_winning_big() -> None:
    """No message should be sent while the position doesn't clear the threshold."""
    board = board_at_move(MIN_MOVE_NUMBER_FOR_TAUNT + 1)
    engine = FakeEngine(scores=[chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE)])
    li = FakeLichess()
    conversation = Conversation(FakeGame(), engine, li, "test-version", [])  # type: ignore[arg-type]

    conversation.maybe_send_taunt(board)
    assert conversation.has_taunted is False
    assert len(li.chat_calls) == 0
