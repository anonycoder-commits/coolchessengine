"""Tests for the PERSONALITY FEATURE: the one-time material-lead taunt in lib/conversation.py."""

import chess
import chess.engine
from lib.conversation import Conversation, is_winning_big, MIN_MOVE_NUMBER_FOR_TAUNT, TAUNT_SCORE_THRESHOLD_CP


class FakeEngine:
    """Minimal stand-in for EngineWrapper.

    is_winning_big only ever reads .last_move_search_score (the real per-search score,
    None when the move came from book/egtb or the search carried no score). .scores is
    kept here only to prove the taunt does NOT read it -- that list pads missing scores
    with a Mate(1) sentinel that once caused false-positive taunts.
    """

    def __init__(self, last_move_search_score: chess.engine.PovScore | None,
                 scores: list[chess.engine.PovScore] | None = None) -> None:
        """Store the fake score state."""
        self.last_move_search_score = last_move_search_score
        self.scores = scores if scores is not None else []


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


def cp_score(cp: int) -> chess.engine.PovScore:
    """A white-POV centipawn score."""
    return chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE)


# --- is_winning_big ---

def test_is_winning_big_false_with_no_search_score() -> None:
    """No search has produced a score yet (book/egtb move or fresh game) -> never a taunt."""
    engine = FakeEngine(last_move_search_score=None)
    board = board_at_move(MIN_MOVE_NUMBER_FOR_TAUNT + 1)
    assert is_winning_big(engine, board) is False


def test_is_winning_big_false_before_the_move_gate_even_with_a_huge_score() -> None:
    """A huge score in the opening shouldn't fire -- avoids early eval/book noise."""
    board = chess.Board()  # fullmove_number == 1
    engine = FakeEngine(last_move_search_score=cp_score(2000))
    assert is_winning_big(engine, board) is False


def test_is_winning_big_true_above_threshold_past_the_move_gate() -> None:
    """Past the move gate, a score clearly above the threshold should fire."""
    board = board_at_move(MIN_MOVE_NUMBER_FOR_TAUNT + 1)
    engine = FakeEngine(last_move_search_score=cp_score(TAUNT_SCORE_THRESHOLD_CP + 50))
    assert is_winning_big(engine, board) is True


def test_is_winning_big_false_below_threshold_past_the_move_gate() -> None:
    """Past the move gate, a modest score should NOT fire."""
    board = board_at_move(MIN_MOVE_NUMBER_FOR_TAUNT + 1)
    engine = FakeEngine(last_move_search_score=cp_score(TAUNT_SCORE_THRESHOLD_CP - 400))
    assert is_winning_big(engine, board) is False


def test_is_winning_big_true_for_a_mate_score() -> None:
    """A found mate for us must count as a big lead -- exercises the mate_score fallback
    (bare .score() returns None for a mate score; the code must never call it bare)."""
    board = board_at_move(MIN_MOVE_NUMBER_FOR_TAUNT + 1)
    engine = FakeEngine(last_move_search_score=chess.engine.PovScore(chess.engine.Mate(4), chess.WHITE))
    assert is_winning_big(engine, board) is True


def test_is_winning_big_ignores_the_mate1_sentinel_in_scores() -> None:
    """REGRESSION (observed live): a search whose result carried no score pads
    engine.scores with a Mate(1) sentinel (~+100000cp, kept for the draw/resign logic).
    last_move_search_score is None in that case, and the taunt must NOT fire -- it
    previously read scores[-1] and taunted at random moments (at evals of +0.5)."""
    board = board_at_move(MIN_MOVE_NUMBER_FOR_TAUNT + 1)
    sentinel = chess.engine.PovScore(chess.engine.Mate(1), chess.WHITE)
    engine = FakeEngine(last_move_search_score=None, scores=[sentinel])
    assert is_winning_big(engine, board) is False


def test_is_winning_big_ignores_stale_scores_after_a_book_move() -> None:
    """A book/egtb move runs no search: play_move resets the field to None even though
    a big score from an earlier search is still sitting in engine.scores."""
    board = board_at_move(MIN_MOVE_NUMBER_FOR_TAUNT + 1)
    engine = FakeEngine(last_move_search_score=None, scores=[cp_score(900)])
    assert is_winning_big(engine, board) is False


# --- Conversation.maybe_send_taunt ---

def test_maybe_send_taunt_fires_once_and_then_stays_silent() -> None:
    """The guard must allow exactly one taunt per Conversation instance."""
    board = board_at_move(MIN_MOVE_NUMBER_FOR_TAUNT + 1)
    engine = FakeEngine(last_move_search_score=cp_score(TAUNT_SCORE_THRESHOLD_CP + 100))
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
    engine = FakeEngine(last_move_search_score=cp_score(0))
    li = FakeLichess()
    conversation = Conversation(FakeGame(), engine, li, "test-version", [])  # type: ignore[arg-type]

    conversation.maybe_send_taunt(board)
    assert conversation.has_taunted is False
    assert len(li.chat_calls) == 0
