"""Manim visual argument: a fresh quick-API arithmetic challenge.

An illustrative animation, not a timed capture of a live service.
"""

from manim import (
    Scene,
    Text,
    VGroup,
    RoundedRectangle,
    Line,
    Dot,
    FadeIn,
    FadeOut,
    Create,
    MoveAlongPath,
    LEFT,
    RIGHT,
    UP,
    DOWN,
)


class FreshChallenge(Scene):
    def construct(self):
        self.camera.background_color = "#071b1b"
        ink = "#f3f0e5"
        teal = "#1cd6ba"
        brass = "#b89a6a"
        label = Text(
            "QUICK API  /  FULL DIFFICULTY", font="Helvetica", font_size=23, color=teal
        ).move_to(UP * 2.35 + LEFT * 1.65)
        title = Text(
            "A fresh arithmetic challenge", font="Helvetica", font_size=40, color=ink
        ).move_to(UP * 1.6)
        panel = RoundedRectangle(
            width=10.7,
            height=2.35,
            corner_radius=0.12,
            stroke_color=brass,
            stroke_width=1.1,
            fill_color="#102a28",
            fill_opacity=0.65,
        )
        expression = Text(
            "48 213 907  +  71 305 118", font="Helvetica", font_size=49, color=ink
        ).move_to(UP * 0.3)
        result = Text(
            "119 519 025", font="Helvetica", font_size=43, color=teal
        ).move_to(DOWN * 0.65)
        pulse = Dot(radius=0.055, color=ink).move_to(LEFT * 5.25 + DOWN * 1.65)
        rail = Line(
            LEFT * 5.25 + DOWN * 1.65,
            RIGHT * 5.25 + DOWN * 1.65,
            color=brass,
            stroke_width=1.5,
        )
        note = Text(
            "A half-second response window", font="Helvetica", font_size=25, color=ink
        ).move_to(DOWN * 2.25)
        self.play(
            FadeIn(label, shift=UP * 0.08), FadeIn(title, shift=UP * 0.1), run_time=1
        )
        self.play(Create(panel), FadeIn(expression, shift=UP * 0.12), run_time=1.2)
        self.wait(2)
        self.play(FadeIn(result, shift=UP * 0.08), run_time=0.7)
        self.wait(1.4)
        self.play(Create(rail), FadeIn(note), FadeIn(pulse), run_time=0.8)
        self.play(MoveAlongPath(pulse, rail), run_time=0.5)
        self.wait(2)
        self.play(
            FadeOut(expression),
            FadeOut(result),
            FadeOut(note),
            FadeOut(rail),
            FadeOut(pulse),
            run_time=0.7,
        )
        a = Text("GENERATED INPUT", font="Helvetica", font_size=26, color=ink).move_to(
            LEFT * 3.15
        )
        b = Text("EXACT RESPONSE", font="Helvetica", font_size=26, color=ink).move_to(
            RIGHT * 3.15
        )
        connector = Line(LEFT * 1.7, RIGHT * 1.7, color=teal, stroke_width=2)
        points = VGroup(
            *[
                Dot(radius=0.045, color=brass).move_to(LEFT * (1.65 - i * 0.3))
                for i in range(12)
            ]
        )
        self.play(FadeIn(a), FadeIn(b), Create(connector), run_time=0.8)
        self.play(FadeIn(points), run_time=0.7)
        self.wait(3)
        self.play(
            FadeOut(points),
            FadeOut(a),
            FadeOut(b),
            FadeOut(connector),
            FadeOut(panel),
            FadeOut(title),
            FadeOut(label),
            run_time=0.7,
        )
