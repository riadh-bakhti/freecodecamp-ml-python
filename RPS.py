# Rock Paper Scissors solution.
#
# The player keeps several "predictors" of the opponent's next move:
#   - Markov chains of order 1..5 built on the opponent's own history
#   - models of the three adaptive bots (they react to *our* history):
#       kris    -> counters our previous move
#       mrugesh -> counters our most frequent move in the last 10
#       abbey   -> counters its order-2 prediction of our next move
# Every predictor is scored on recent games and the best one is used.
# We then play the move that beats the predicted opponent move.

COUNTER = {"R": "P", "P": "S", "S": "R"}  # move that beats the key
MOVES = ("R", "P", "S")

MARKOV_ORDERS = (1, 2, 3, 4, 5)
WINDOW = 80  # number of recent games used to score a predictor


def _reset(opponent_history, my_history, scores, last_predictions):
    opponent_history.clear()
    my_history.clear()
    scores.clear()
    last_predictions.clear()


def _markov_predict(history, order):
    """Most frequent continuation of the last `order` moves."""
    if len(history) < order or (order and history[-1] == ""):
        return None
    pattern = tuple(history[-order:])
    counts = {}
    last_seen = {}
    for i in range(order, len(history)):
        if tuple(history[i - order:i]) == pattern:
            nxt = history[i]
            counts[nxt] = counts.get(nxt, 0) + 1
            last_seen[nxt] = i
    if not counts:
        return None
    # Highest count, most recent occurrence as tie-break.
    return max(counts, key=lambda m: (counts[m], last_seen[m]))


def _kris_predict(my_history):
    if not my_history:
        return "P"  # kris answers the initial "R" with P
    return COUNTER[my_history[-1]]


def _mrugesh_predict(my_history):
    last_ten = my_history[-10:]
    if not last_ten:
        return "R"  # mrugesh falls back to "S", we beat S with R
    most_frequent = max(MOVES, key=lambda m: (last_ten.count(m), -MOVES.index(m)))
    return COUNTER[most_frequent]


def _abbey_predict(my_history):
    """Replicate abbey's order-2 model of our moves."""
    if not my_history:
        return COUNTER["R"]
    play_order = {}
    for i in range(2, len(my_history) + 1):
        pair = my_history[i - 2] + my_history[i - 1]
        play_order[pair] = play_order.get(pair, 0) + 1
    last = my_history[-1]
    candidates = {last + m: play_order.get(last + m, 0) for m in MOVES}
    predicted_our_move = max(candidates, key=candidates.get)[-1]
    return COUNTER[predicted_our_move]


def _all_predictions(opponent_history, my_history):
    predictions = {}
    for order in MARKOV_ORDERS:
        predictions[f"markov{order}"] = _markov_predict(opponent_history, order)
    predictions["kris"] = _kris_predict(my_history)
    predictions["mrugesh"] = _mrugesh_predict(my_history)
    predictions["abbey"] = _abbey_predict(my_history)
    return predictions


def player(prev_play, opponent_history=[], my_history=[], scores={},
           last_predictions={}):
    # A new match always starts with prev_play == "".
    if prev_play == "":
        _reset(opponent_history, my_history, scores, last_predictions)
    else:
        opponent_history.append(prev_play)
        # Score the prediction that was made on the previous turn.
        for name, prediction in last_predictions.items():
            if prediction is not None:
                history = scores.setdefault(name, [])
                history.append(1 if prediction == prev_play else 0)
                if len(history) > WINDOW:
                    del history[0]

    predictions = _all_predictions(opponent_history, my_history)
    last_predictions.clear()
    last_predictions.update(predictions)

    def score(name):
        history = scores.get(name, [])
        # Optimistic prior so untested predictors get a chance.
        return (sum(history) + 1) / (len(history) + 2)

    best = max(predictions, key=lambda name: (score(name), predictions[name] is not None))

    if predictions.get(best) is None:
        # No usable pattern yet: beat the opponent's most common move.
        if opponent_history:
            expected = max(MOVES, key=opponent_history.count)
        else:
            expected = "R"
    else:
        expected = predictions[best]

    move = COUNTER[expected]
    my_history.append(move)
    return move
