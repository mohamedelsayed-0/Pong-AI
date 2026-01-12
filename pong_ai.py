"""
submission for the ESC180 pong ai vs AI tournament :)

idea behind code summarized:
- attempt to estimate the ball's velocity from frame to frame
- predict where it will cross my wall's x pos using a bounce model (phy180 :/)
- when the ball is coming toward me, line up aggressively with where it predicts it will go
- when the ball is going away, move to a good resting position instead of random chasing
- use a small buffer/dead zone so the wall doesn't break when its optimizing for small amounts of pixels
"""

# if you want to test this code, go to the variables score_to_win and edit it to less then 1000 and then change the dispplay vairable to 1

# global state to remember previous ball center and velocity
_last_ball_cx = None
_last_ball_cy = None
_last_vx = None
_last_vy = None


def _predict_ball_y(ball_cy, vy, t, table_height):
    """
    - pretend the ball moving is moving in its currently trajectory in a straight line infinitely, like an intertial frame or FBD
    - use % to fold that path back into the table height
    - this instantly recreates the correct bounce point without actually simulating each frame
    - inspiration for this idea was feynman technique 
    """
    if t <= 0 or abs(vy) < 1e-9:
        # no time has passed, or the ball has basically no vertical speed
        return ball_cy

    # where the ball would be if there were no walls
    y_wallless = ball_cy + vy * t
    period = 2.0 * table_height  # one full up + down

    # fold this into a single up+down cycle
    y_mod = y_wallless % period

    # if we are in the "coming back down" half, reflect
    if y_mod > table_height:
        return period - y_mod
    else:
        return y_mod


def pong_ai(paddle_frect, other_paddle_frect, ball_frect, table_size):
    """
    main ai function returns "up" or "down" for wall movement

     idea:
    - compute the center of the wall and of the ball
    - estimate ball velocity using smoothing and a basic outlier check
    - check if the ball is moving toward us or away from us
    - if it is coming toward us with positive time-to-impact:
        predict where it will cross our x and move there
        if it is very close and fast, bias slightly toward the wall's edges to create stronger angles
      else:
        pick a smart waiting position (not just sitting at center)
    - use an adaptive tolerance (buffer zone) so the wall doesn't shake
      when the difference is tiny
    """
    global _last_ball_cx, _last_ball_cy, _last_vx, _last_vy

    # wall info top-left position and size
    (px, py) = paddle_frect.pos
    (pw, ph) = paddle_frect.size

    # ball info: top-left position and size
    (bx, by) = ball_frect.pos
    (bw, bh) = ball_frect.size

    table_w, table_h = table_size

    # compute centers of wall and ball
    paddle_cx = px + pw / 2.0
    paddle_cy = py + ph / 2.0

    ball_cx = bx + bw / 2.0
    ball_cy = by + bh / 2.0

    # instantaneous velocity based on last frame
    if _last_ball_cx is None or _last_ball_cy is None:
        # no history yet
        inst_vx = 0.0
        inst_vy = 0.0
    else:
        inst_vx = ball_cx - _last_ball_cx
        inst_vy = ball_cy - _last_ball_cy

    # smooth the velocity using an exponential moving average
    # this makes prediction less noisy, especially around bounces
    if _last_vx is None or _last_vy is None:
        # no previous velocity estimate yet
        vx = inst_vx
        vy = inst_vy
    else:
        # check how different the new velocity is compared to the old one
        # big jumps usually happen when the ball bounces
        vx_change = abs(inst_vx - _last_vx)
        vy_change = abs(inst_vy - _last_vy)
        
        if vx_change > 2.0 or vy_change > 2.0:
            # big sudden change ->ikely a bounce ->trust the new value more
            alpha = 0.6
        else:
            # normal frames-> value the new and old moderately
            alpha = 0.35
            
        vx = (1.0 - alpha) * _last_vx + alpha * inst_vx
        vy = (1.0 - alpha) * _last_vy + alpha * inst_vy

    # update stored state for next call
    _last_ball_cx = ball_cx
    _last_ball_cy = ball_cy
    _last_vx = vx
    _last_vy = vy

    # figure out if we are the left wall or the right wall
    is_left = paddle_cx < table_w / 2.0

    # check if the ball is moving toward our side
    if is_left:
        toward_me = vx < 0
    else:
        toward_me = vx > 0

    # time (in frames) until ball reaches our x-position
    if abs(vx) < 1e-9:
        # basically vertical movement only
        t_to_paddle = float('inf')
    else:
        t_to_paddle = (paddle_cx - ball_cx) / vx

    # makes base target center of the playfield
    target_cy = table_h / 2.0
    
    # horizontal distance to ball and its speed
    dist_x = abs(paddle_cx - ball_cx)
    ball_speed = (vx**2 + vy**2)**0.5

    # default case: ball is coming toward us and hasn't passed yet
    if toward_me and t_to_paddle > 0:
        # predict where the ball will be when it reaches our x
        predicted_y = _predict_ball_y(ball_cy, vy, t_to_paddle, table_h)
        target_cy = predicted_y

        # make sure wall center stays fully inside table
        half_ph = ph / 2.0
        if target_cy < half_ph:
            target_cy = half_ph
        elif target_cy > table_h - half_ph:
            target_cy = table_h - half_ph
            
        # if the ball is very close and moving pretty fast than
        # shift slightly toward wall edges to create angled returns
        if dist_x < table_w * 0.15 and abs(ball_speed) > 1.5:
            if predicted_y < table_h / 2.0:
                # ball in top half → aim a bit higher on the wall
                target_cy = max(target_cy - ph * 0.15, half_ph)
            else:
                # ball in bottom half → aim a bit lower on the wall
                target_cy = min(target_cy + ph * 0.15, table_h - half_ph)
    else:
        # ball is moving away from us or time-to-impact isn't positive
        # here we try to position well for the next rally instead of random chasing

        # if the ball is clearly on the opponent's side
        if (is_left and ball_cx > table_w / 2.0) or (not is_left and ball_cx < table_w / 2.0):
            # assume most opponents line up roughly with the ball center
            expected_return_y = ball_cy

            # pick a spot between center and expected return position
            target_cy = expected_return_y * 0.4 + (table_h / 2.0) * 0.6

    # tolerance
    # smaller when the ball is close and coming toward us (more precise),
    # bigger when it's far or moving away (less twitchy)
    if toward_me and dist_x < table_w * 0.25:
        # close and approaching so we want tight control
        tolerance = ph * 0.02
    elif toward_me and dist_x < table_w * 0.5:
        # medium distance and approaching
        tolerance = ph * 0.05
    else:
        # far away or not approaching
        tolerance = ph * 0.12

    # minimum tolerance so we don't freak out over sub-pixel differences
    tolerance = max(tolerance, 0.8)
    
    # how far we are from where we want to be
    diff = target_cy - paddle_cy

    # (duplicate line kept to match original code behavior exactly)
    diff = target_cy - paddle_cy

    # main movement decision
    if diff > tolerance:
        # target is below us
        return "down"
    elif diff < -tolerance:
        # target is above us
        return "up"
    else:
        # in the "dead zone" where we are close enough to target
        # behavior here depends on how close and whether ball is coming

        if toward_me and dist_x < table_w * 0.4:
            # ball is coming and somewhat close → try not to over-move
            if abs(diff) > 0.5:
                # small adjustment if needed
                return "down" if diff > 0 else "up"
            else:
                # basically good enough, pick a stable direction
                return "up"  # arbitrary but consistent
        else:
            # ball far or moving away → gently drift toward table center
            center_offset = paddle_cy - table_h / 2.0
            if abs(center_offset) > ph * 0.05:
                # if we are noticeably off-center, move back a bit
                return "up" if center_offset > 0 else "down"
            else:
                # super close to ideal → stay in place to avoid jitter
                return "up" if diff < 0 else "down"
