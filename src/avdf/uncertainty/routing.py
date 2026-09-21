

def route(prob, uncertainty, tau=0.85, u_max=0.5):
    """Auto-accept only if confidence >= tau and uncertainty <= u_max; otherwise send to human review."""
    conf, pred = prob.max(-1)
    review = (conf < tau) | (uncertainty > u_max)
    return pred, conf, review
