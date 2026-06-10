from manim import *
import os

os.environ["FFMPEG_BINARY"] = r"C:\ProgramData\anaconda3\Library\bin\ffmpeg.exe"

class VectorAddition(Scene):
    def construct(self):
        plane = NumberPlane()
        self.add(plane)

        v1 = Vector([2, 3], color=BLUE)
        v2 = Vector([3, 1], color=GREEN).shift(v1.get_end())
        result = Vector([5, 4], color=YELLOW)

        label1 = Text("v1 = (2,3)", color=BLUE).next_to(v1, LEFT)
        label2 = Text("v2 = (3,1)", color=GREEN).next_to(v2, RIGHT)
        label_res = Text("v1 + v2 = (5,4)", color=YELLOW).to_edge(UP)

        self.play(GrowArrow(v1))
        self.play(Write(label1))
        self.play(GrowArrow(v2))
        self.play(Write(label2))
        self.play(GrowArrow(result))
        self.play(Write(label_res))

        self.wait(3)
