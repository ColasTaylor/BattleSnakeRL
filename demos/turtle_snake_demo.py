import random
import time
import turtle
from turtle import Turtle, Screen
from typing import List, Optional, Tuple

turtle.colormode(255)
MOVE_DIST = 20

# ===================== SNAKE =====================


class Snake:
    """Turtle-based snake with body segments and simple movement."""

    def __init__(
        self,
        start_pos: Tuple[int, int] = (0, 0),
        base_color: Tuple[int, int, int] = (180, 60, 200),
        head_color: str = "pink",
    ) -> None:
        self.all_tuts: List[Turtle] = []
        self.base_color = base_color
        self.create_snake(15, start_pos)
        self.head = self.all_tuts[0]
        self.head.color(head_color)

    def add_part(self) -> None:
        idx = len(self.all_tuts)
        base_r, base_g, base_b = self.base_color

        # make a little gradient along the body
        factor = 5
        r = max(0, base_r - idx * factor)
        g = max(0, base_g - idx * factor)
        b = max(0, base_b - idx * factor)

        seg = Turtle(shape="square")
        seg.color((r, g, b))
        seg.penup()
        # temporary spot; will be repositioned on first move
        seg.goto(1000, 1000)
        self.all_tuts.append(seg)

    def create_snake(self, count: int, start_pos: Tuple[int, int]) -> None:
        x0, y0 = start_pos
        for i in range(count):
            self.add_part()
            self.all_tuts[i].goto(x0 - i * 20, y0)

    def up(self) -> None:
        if self.head.heading() != 270:
            self.head.setheading(90)

    def down(self) -> None:
        if self.head.heading() != 90:
            self.head.setheading(270)

    def left(self) -> None:
        if self.head.heading() != 0:
            self.head.setheading(180)

    def right(self) -> None:
        if self.head.heading() != 180:
            self.head.setheading(0)

    def move(self) -> None:
        for num in range(len(self.all_tuts) - 1, 0, -1):
            x = self.all_tuts[num - 1].xcor()
            y = self.all_tuts[num - 1].ycor()
            self.all_tuts[num].goto(x, y)
        self.head.forward(MOVE_DIST)

    def wrap(self, limit: int = 300) -> None:
        # same wrap-around logic as before
        x = self.head.xcor()
        y = self.head.ycor()
        wrapped = False
        if x > limit or x < -limit:
            x = -x
            wrapped = True
        if y > limit or y < -limit:
            y = -y
            wrapped = True
        if wrapped:
            self.head.goto(x, y)

    def hit_self(self) -> bool:
        for segment in self.all_tuts[1:]:
            if self.head.distance(segment) < 10:
                return True
        return False

    def head_collides_with_snake(self, other_snake: "Snake") -> bool:
        # head-to-head
        if self.head.distance(other_snake.head) < 10:
            return True
        # head-to-body
        for seg in other_snake.all_tuts[1:]:
            if self.head.distance(seg) < 10:
                return True
        return False


# ===================== FOOD =====================


class Food(Turtle):
    def __init__(self) -> None:
        super().__init__()
        self.shape("circle")
        self.penup()
        self.color("yellow")
        self.shapesize(stretch_wid=0.5, stretch_len=0.5)
        self.speed("fastest")
        self.refresh()

    def refresh(self) -> None:
        x_c = random.randint(-280, 280)
        y_c = random.randint(-280, 280)
        self.goto(x_c, y_c)


# ===================== SCOREBOARD (2 players) =====================

ALIGNMENT = "center"
FONT = ("Arial", 18, "bold")


class Scoreboard(Turtle):
    def __init__(self) -> None:
        super().__init__()
        self.score1 = 0
        self.score2 = 0
        self.penup()
        self.color("white")
        self.hideturtle()
        self.goto(0, 260)
        self.write_scores()

    def write_scores(self) -> None:
        self.clear()
        self.write(
            f"P1: {self.score1}    P2: {self.score2}", align=ALIGNMENT, font=FONT
        )

    def gain_score(self, player: int) -> None:
        if player == 1:
            self.score1 += 1
        else:
            self.score2 += 1
        self.write_scores()

    def game_over(self, msg: str) -> None:
        self.goto(0, 0)
        self.write(msg, align=ALIGNMENT, font=("Arial", 22, "bold"))


# ===================== RL ACTION SPACE HELPERS =====================

# We’ll use 4 discrete actions for now: absolute directions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

NUM_ACTIONS = 4


def apply_action(snake: Snake, action: int):
    """Map discrete action to the snake's direction change."""
    if action == ACTION_UP:
        snake.up()
    elif action == ACTION_DOWN:
        snake.down()
    elif action == ACTION_LEFT:
        snake.left()
    elif action == ACTION_RIGHT:
        snake.right()
    # The Snake.up/down/left/right methods already prevent 180° turns
    # so we don't need extra logic here.


def get_state(snake: Snake, food: Turtle) -> List[float]:
    """Very simple state representation for now:
    [head_x, head_y, food_x, food_y] normalized to [-1, 1] range.

    Later we’ll replace this with a grid-based observation.
    """
    limit = 300.0  # roughly your playfield half-size
    hx, hy = snake.head.xcor(), snake.head.ycor()
    fx, fy = food.xcor(), food.ycor()

    return [
        hx / limit,
        hy / limit,
        fx / limit,
        fy / limit,
    ]


# ===================== RL ENVIRONMENT WRAPPER =====================


class SnakeEnv:
    """Simple single-snake RL environment wrapped around your existing Snake + Food.

    API roughly matches OpenAI Gym:
        obs = env.reset()
        obs, reward, done, info = env.step(action)
        env.render()
    """

    def __init__(self, render_mode: bool = True, max_steps: int = 500) -> None:
        self.render_mode = render_mode
        self.max_steps = max_steps

        # Create the Turtle screen once
        self.screen = Screen()
        self.screen.setup(width=600, height=600)
        self.screen.bgcolor("black")
        self.screen.title("Snake RL Env")
        # Manual screen updates
        self.screen.tracer(0)

        # Placeholders; real objects created in reset()
        self.snake: Optional[Snake] = None
        self.food: Optional[Food] = None
        self.steps = 0
        self.score = 0

    def reset(self) -> List[float]:
        """Start a brand new episode and return initial observation."""
        # Clear old turtles if they exist
        if self.snake is not None:
            for seg in self.snake.all_tuts:
                seg.hideturtle()
        if self.food is not None:
            self.food.hideturtle()

        # Recreate snake and food
        self.snake = Snake(start_pos=(0, 0))
        self.food = Food()

        self.steps = 0
        self.score = 0

        obs = get_state(self.snake, self.food)

        if self.render_mode:
            self.render()

        return obs

    def step(self, action: int):
        """Advance the game by one time step given an action.

        Returns:
            obs: next observation
            reward: float
            done: bool
            info: dict (empty for now)
        """
        if self.snake is None or self.food is None:
            raise RuntimeError("Call reset() before step().")

        self.steps += 1

        # Small negative reward each step to encourage shorter paths
        reward = -0.01
        done = False
        info = {}

        # 1) Apply action (change direction) and move the snake
        apply_action(self.snake, action)
        self.snake.move()
        self.snake.wrap(limit=280)

        # 2) Check food collision
        if self.snake.head.distance(self.food) < 15:
            self.food.refresh()
            self.snake.add_part()
            self.score += 1
            reward += 1.0  # eating food is good

        # 3) Check self collision => game over
        if self.snake.hit_self():
            reward -= 1.0
            done = True

        # 4) Optional: time-limit the episode
        if self.steps >= self.max_steps:
            done = True

        # 5) Build next observation
        obs = get_state(self.snake, self.food)

        if self.render_mode:
            self.render()

        return obs, reward, done, info

    def render(self) -> None:
        """Update the Turtle window."""
        self.screen.update()

    def close(self) -> None:
        """Close the Turtle window."""
        self.screen.bye()


# ===================== MAIN GAME (HUMAN) =====================


def play_human() -> None:
    screen = Screen()
    screen.setup(width=600, height=600)
    screen.bgcolor("black")
    screen.title("Multiplayer Snake")
    screen.tracer(0)

    # Player 1 (arrows) at top
    snake1 = Snake(start_pos=(0, 40), base_color=(180, 60, 200), head_color="pink")
    # Player 2 (WASD) at bottom
    snake2 = Snake(start_pos=(0, -40), base_color=(60, 200, 180), head_color="cyan")

    food = Food()
    scoreboard = Scoreboard()

    screen.listen()
    # P1 controls
    screen.onkey(snake1.up, "Up")
    screen.onkey(snake1.down, "Down")
    screen.onkey(snake1.left, "Left")
    screen.onkey(snake1.right, "Right")

    # P2 controls (WASD)
    screen.onkey(snake2.up, "w")
    screen.onkey(snake2.down, "s")
    screen.onkey(snake2.left, "a")
    screen.onkey(snake2.right, "d")

    game_on = True
    p1_alive = True
    p2_alive = True

    while game_on:
        screen.update()
        # speed: you can keep it fixed for multiplayer
        time.sleep(0.1)

        if p1_alive:
            snake1.move()
            snake1.wrap()
        if p2_alive:
            snake2.move()
            snake2.wrap()

        # food collisions
        if p1_alive and snake1.head.distance(food) < 15:
            food.refresh()
            scoreboard.gain_score(1)
            snake1.add_part()

        if p2_alive and snake2.head.distance(food) < 15:
            food.refresh()
            scoreboard.gain_score(2)
            snake2.add_part()

        # collisions
        # 1) self-collision
        if p1_alive and snake1.hit_self():
            p1_alive = False
            scoreboard.game_over("Player 1 crashed! Player 2 wins!")

        if p2_alive and snake2.hit_self():
            p2_alive = False
            scoreboard.game_over("Player 2 crashed! Player 1 wins!")

        # 2) head vs opponent
        if p1_alive and snake1.head_collides_with_snake(snake2):
            p1_alive = False
            scoreboard.game_over("Player 1 crashed into Player 2! Player 2 wins!")

        if p2_alive and snake2.head_collides_with_snake(snake1):
            p2_alive = False
            scoreboard.game_over("Player 2 crashed into Player 1! Player 1 wins!")

        # if both die same frame (head-to-head), show draw
        if not p1_alive and not p2_alive:
            scoreboard.game_over("Both crashed! It's a draw!")
            game_on = False

        # stop if someone lost
        if not p1_alive or not p2_alive:
            game_on = False

    screen.exitonclick()


# ===================== MAIN (RL DEMO) =====================


def main() -> None:
    """Tiny demo: run one episode with a random policy."""
    env = SnakeEnv(render_mode=True, max_steps=200)
    obs = env.reset()

    done = False
    total_reward = 0.0

    while not done:
        # For now, choose random actions.
        # Later, your PPO / DQN agent will go here.
        action = random.randint(0, NUM_ACTIONS - 1)

        obs, reward, done, info = env.step(action)
        total_reward += reward

        time.sleep(0.1)  # control visual speed; remove for fast training

    print("Episode finished, total reward:", total_reward)
    env.screen.exitonclick()


if __name__ == "__main__":
    # Choose what to run when you execute `python main.py`
    main()  # RL (random agent) demo
    # play_human()   # classic keyboard-controlled game
