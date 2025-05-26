import tkinter as tk
from tkinter import Canvas
import numpy as np
from scipy.interpolate import make_interp_spline

"""
Win Probability Overlay Module

This module provides a floating graphical overlay and accompanying probability graph 
for visualizing win probability during League of Legends matches.
"""

class WinProbabilityOverlay:
    def __init__(self, predictor):
        self.predictor = predictor
        self.predictor.add_viewer(self)

        # Main overlay window
        self.root = tk.Tk()
        self.root.attributes("-topmost", True)
        self.root.overrideredirect(True)
        self.root.attributes("-alpha", 0.75)
        self.root.geometry("516x35+705+90")  
        self.root.configure(bg='#100719')
        
        self.canvas = Canvas(self.root, width=516, height=35, bg='#100719', highlightthickness=0)
        self.canvas.pack()

        self.blue_color = "#01e4be"
        self.red_color = "#fd5018"
        
        self.current_blue_width = 0
        self.current_red_width = 506  
        self.hover_glow_id = None

        self.calculating = True
        self.calculating_dot_count = 0
        self.root.after(500, self.animate_calculating_text)

        # Control buttons
        self.minimize_button = tk.Button(
            self.root, text="—", command=self.minimize,
            font=("Segoe UI", 10, 'bold'), bg="#222", fg="white",
            activebackground="#444", activeforeground="white",
            bd=0, highlightthickness=0, relief='flat'
        )
        self.exit_button = tk.Button(
            self.root, text="✕", command=self.exit_program,
            font=("Segoe UI", 10, 'bold'), bg="#900", fg="white",
            activebackground="#c00", activeforeground="white",
            bd=0, highlightthickness=0, relief='flat'
        )
        self.graph_toggle_button = tk.Button(
            self.root, text="📈", command=self.toggle_graph,
            font=("Segoe UI", 10, 'bold'), bg="#444", fg="white",
            activebackground="#666", activeforeground="white",
            bd=0, highlightthickness=0, relief='flat'
        )

        self.minimize_button.bind("<Enter>", lambda e: self.show_glow(425, 6))
        self.minimize_button.bind("<Leave>", lambda e: self.hide_glow())
        self.graph_toggle_button.bind("<Enter>", lambda e: self.show_glow(455, 6))
        self.graph_toggle_button.bind("<Leave>", lambda e: self.hide_glow())
        self.exit_button.bind("<Enter>", lambda e: self.show_glow(485, 6))
        self.exit_button.bind("<Leave>", lambda e: self.hide_glow())

        self.minimize_button.place_forget()
        self.exit_button.place_forget()
        self.graph_toggle_button.place_forget()

        self.root.bind("<Enter>", self.show_buttons)
        self.root.bind("<Leave>", self.hide_buttons)

        # Graph window
        self.graph_window = tk.Toplevel(self.root)
        self.graph_window.withdraw()  
        self.graph_window.geometry("725x252+600+576") 
        self.graph_window.overrideredirect(True)
        self.graph_window.attributes("-topmost", True)
        self.graph_canvas = Canvas(self.graph_window, width=725, height=252, bg='#100719', highlightthickness=0)
        self.graph_canvas.pack()
        
        self.graph_close_button = tk.Button(
            self.graph_window,
            text="✕",  
            command=self.toggle_graph,
            font=("Malgun Gothic", 10, 'bold'),
            bg="#222",
            fg="white",
            activebackground="#c00",
            activeforeground="white",
            bd=0,
            highlightthickness=0
        )

        self.graph_close_button.place(x=700, y=5, width=20, height=20)

        self.times = []
        self.probs = []

        #restore button
        self.restore_window = tk.Toplevel(self.root)
        self.restore_window.geometry("60x25+936+90")  
        self.restore_window.overrideredirect(True)
        self.restore_window.attributes("-topmost", True)
        self.restore_window.configure(bg="#100719", bd=0, highlightthickness=0)
        self.restore_window.attributes("-alpha", 0.75)
        self.restore_button = tk.Button(self.restore_window, text="WIN %", command=self.restore, font=("Malgun Gothic", 10, "bold"), bg="#100719", fg="white")
        self.restore_button.pack(fill=tk.BOTH, expand=True)
        self.restore_window.bind("<ButtonPress-1>", self.start_restore_move)
        self.restore_window.bind("<B1-Motion>", self.do_restore_move)
        self.restore_window.withdraw()


    def animate_calculating_text(self):
        if self.calculating:
            self.canvas.delete("calculating_text")  
            text = "Calculating" + "." * (self.calculating_dot_count % 4)
            self.canvas.create_text(258, 17, text=text, fill="white", font=('Malgun Gothic', 11, 'bold'), tags="calculating_text")
            self.calculating_dot_count += 1
            self.root.after(500, self.animate_calculating_text)


    def show_glow(self, x, y):
        if self.hover_glow_id is not None:
            self.canvas.delete(self.hover_glow_id)
        self.hover_glow_id = self.canvas.create_oval(
            x-2, y-2, x+22, y+22,
            fill="#01e4be", outline="",
            stipple="gray50"  
        )


    def hide_glow(self):
        if self.hover_glow_id is not None:
            self.canvas.delete(self.hover_glow_id)
            self.hover_glow_id = None


    def start_restore_move(self, event):
        self._restore_offset_x = event.x_root - self.restore_window.winfo_x()
        self._restore_offset_y = event.y_root - self.restore_window.winfo_y()
        self.is_dragging_restore = False  # Track if user is dragging


    def do_restore_move(self, event):

        self.is_dragging_restore = True
        x = event.x_root - self._restore_offset_x
        y = event.y_root - self._restore_offset_y
        self.restore_window.geometry(f"60x25+{x}+{y}")


    def restore(self):
        if not self.is_dragging_restore:
            self.root.deiconify()
            self.restore_window.withdraw()


    def toggle_graph(self):
        if self.graph_window.state() == "withdrawn":
            self.graph_window.deiconify()
        else:
            self.graph_window.withdraw()


    def show_buttons(self, event=None):
        self.minimize_button.place(x=426, y=6, width=20, height=20)
        self.graph_toggle_button.place(x=456, y=6, width=20, height=20)
        self.exit_button.place(x=486, y=6, width=20, height=20)


    def hide_buttons(self, event=None):
        self.minimize_button.place_forget()
        self.exit_button.place_forget()
        self.graph_toggle_button.place_forget()


    def smooth_update(self, target_blue_width, target_red_width, step=10):
        if abs(self.current_blue_width - target_blue_width) < step and abs(self.current_red_width - target_red_width) < step:
            self.current_blue_width = target_blue_width
            self.current_red_width = target_red_width
        else:
            if self.current_blue_width < target_blue_width:
                self.current_blue_width += step
            elif self.current_blue_width > target_blue_width:
                self.current_blue_width -= step
            
            if self.current_red_width < target_red_width:
                self.current_red_width += step
            elif self.current_red_width > target_red_width:
                self.current_red_width -= step
            
            self.root.after(15, self.smooth_update, target_blue_width, target_red_width, step)
        
        self.draw_bars()


    def draw_bars(self):
        self.canvas.delete("all")
        x_start = 5
        bar_height = 25
        
        self.canvas.create_rectangle(x_start, 5, x_start + self.current_blue_width, 5 + bar_height, fill=self.blue_color, outline="")
        self.canvas.create_rectangle(x_start + self.current_blue_width, 5, x_start + self.current_blue_width + self.current_red_width, 5 + bar_height, fill=self.red_color, outline="")
        
        blue_prob = (self.current_blue_width / 506) * 100
        red_prob = 100 - blue_prob
        
        self.canvas.create_text(x_start + self.current_blue_width // 2, 15, text=f"{blue_prob:.2f}%", fill="#100719", font=('Malgun Gothic', 10, 'bold'))
        self.canvas.create_text(x_start + self.current_blue_width + self.current_red_width // 2, 15, text=f"{red_prob:.2f}%", fill="#100719", font=('Malgun Gothic', 10, 'bold'))
    
    def draw_graph(self):
        self.graph_canvas.delete("all")
        
        if len(self.times) < 3:
            return
        
        x_vals = np.array(self.times)
        y_vals = np.array(self.probs)
        x_vals = (x_vals - x_vals.min()) / (x_vals.max() - x_vals.min()) * 725
        
        x_smooth = np.linspace(x_vals.min(), x_vals.max(), 300)
        spline = make_interp_spline(x_vals, y_vals, k=1) 
        y_smooth = spline(x_smooth)
        
        prev_x, prev_y = None, None
        for x, y in zip(x_smooth, y_smooth):
            color = "#01e4be" if y >= 50 else "#fd5018"
            if prev_x is not None:
                self.graph_canvas.create_line(prev_x, 252 - (prev_y / 100 * 252),
                                              x, 252 - (y / 100 * 252),
                                              fill=color, width=2)
            prev_x, prev_y = x, y
        
        self.graph_canvas.create_text(1, 10, text="100%", fill="white", font=("Malgun Gothic", 10, 'bold'), anchor="w")
        self.graph_canvas.create_text(1, 242, text="100%", fill="white", font=("Malgun Gothic", 10, 'bold'), anchor="w")
        self.graph_canvas.create_text(362, 10, text="WIN%", fill="white", font=("Malgun Gothic", 10, 'bold'), anchor="w")
        self.graph_canvas.create_line(0, 126, 725, 126, fill="white", dash=(4, 2))
    

    def update(self, gametime, prob):     
        if self.calculating:
            self.calculating = False

        self.times.append(gametime)
        self.probs.append(prob * 100)
        self.smooth_update(int(prob * 506), 506 - int(prob * 506))
        self.draw_graph()


    def minimize(self):
        self.root.withdraw()
        self.restore_window.deiconify()


    def exit_program(self):
        self.predictor.end()
        self.root.quit()

    
    def run(self):
        self.root.mainloop()
