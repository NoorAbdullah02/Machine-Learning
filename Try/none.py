import turtle

def draw_named_bus(name):
    s = turtle.Screen()
    s.setup(600, 400)

    t = turtle.Turtle()
    t.speed(5)

    # ---- Bus Body ----
    t.penup()
    t.goto(-150, 0)
    t.pendown()
    t.color("green")
    t.begin_fill()
    for _ in range(2):
        t.forward(300)
        t.left(90)
        t.forward(120)
        t.left(90)
    t.end_fill()

    # ---- Bus Name ----
    t.penup()
    t.goto(0, 40)
    t.color("black")
    t.write(name, align="center", font=("Arial", 18, "bold"))

    # ---- Windows ----
    t.color("skyblue")
    for x in [-120, -40, 40, 120]:
        t.penup()
        t.goto(x, 60)
        t.pendown()
        t.begin_fill()
        for _ in range(4):
            t.forward(40)
            t.left(90)
        t.end_fill()

    # ---- Wheels ----
    t.color("black")
    for pos in [-100, 100]:
        t.penup()
        t.goto(pos, -20)
        t.pendown()
        t.begin_fill()
        t.circle(25)
        t.end_fill()

    t.hideturtle()
    s.exitonclick()

draw_named_bus("Abdulla Vai")