"""
line_follower.py
Two McCulloch–Pitts neurons for left-edge following of a thick line
(Task 2). The assignment's robot source was not attached; this module
implements the controller that would be dropped into that source.

Convention
    Sensor = 1 when the sensor sits on the line, 0 when off it.
    Output = 1 iff weighted sum >= 0, else 0.
    Weights format: {{bias1, w11, w21}, {bias2, w12, w22}}
        w11 : left sensor  → output 1
        w21 : right sensor → output 1
        w12 : left sensor  → output 2
        w22 : right sensor → output 2

Intended behaviour (second truth table): follow the LEFT edge.
    (0,0) lost the line          → output (1,0)  search / turn one way
    (0,1) only right on the line → output (0,1)  steer back toward left edge
    (1,0) only left on the line  → output (1,1)  go forward (on the edge)
    (1,1) both on the line       → output (0,1)  steer off the thick line
                                    toward its left edge
"""


def hardlim(s):
    return 1 if s >= 0 else 0


def neuron(bias, w_left, w_right, left, right):
    return hardlim(bias + w_left * left + w_right * right)


# Chosen weights (see report for the linear inequalities)
WEIGHTS = {
    "output1": {"bias": 0.5, "w_left": 1.0, "w_right": -2.0},
    "output2": {"bias": -0.5, "w_left": 1.0, "w_right": 1.0},
}

# Assignment array form
WEIGHTS_ARRAY = (
    (WEIGHTS["output1"]["bias"], WEIGHTS["output1"]["w_left"], WEIGHTS["output1"]["w_right"]),
    (WEIGHTS["output2"]["bias"], WEIGHTS["output2"]["w_left"], WEIGHTS["output2"]["w_right"]),
)


def controller(left_sensor, right_sensor):
    o1 = neuron(WEIGHTS["output1"]["bias"],
                WEIGHTS["output1"]["w_left"],
                WEIGHTS["output1"]["w_right"],
                left_sensor, right_sensor)
    o2 = neuron(WEIGHTS["output2"]["bias"],
                WEIGHTS["output2"]["w_left"],
                WEIGHTS["output2"]["w_right"],
                left_sensor, right_sensor)
    return o1, o2


if __name__ == "__main__":
    print("Weights = {{%.1f, %.1f, %.1f}, {%.1f, %.1f, %.1f}}" % (
        WEIGHTS_ARRAY[0][0], WEIGHTS_ARRAY[0][1], WEIGHTS_ARRAY[0][2],
        WEIGHTS_ARRAY[1][0], WEIGHTS_ARRAY[1][1], WEIGHTS_ARRAY[1][2],
    ))
    print()
    print("L  R  |  O1  O2   sum1   sum2")
    print("-------------------------------")
    target = {
        (0, 0): (1, 0),
        (0, 1): (0, 1),
        (1, 0): (1, 1),
        (1, 1): (0, 1),
    }
    ok = True
    for L, R in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        o1, o2 = controller(L, R)
        s1 = WEIGHTS["output1"]["bias"] + WEIGHTS["output1"]["w_left"] * L + WEIGHTS["output1"]["w_right"] * R
        s2 = WEIGHTS["output2"]["bias"] + WEIGHTS["output2"]["w_left"] * L + WEIGHTS["output2"]["w_right"] * R
        flag = "" if (o1, o2) == target[(L, R)] else "  MISMATCH"
        if flag:
            ok = False
        print("%d  %d  |   %d   %d   %5.1f  %5.1f%s" % (L, R, o1, o2, s1, s2, flag))
    print()
    print("Table match:" , "YES" if ok else "NO")
