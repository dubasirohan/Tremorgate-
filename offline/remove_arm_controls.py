from pathlib import Path

path = Path("flex_dashboard.py")
code = path.read_text()

code = code.replace("ARM_PORT = \"/dev/ttyUSB1\"\n", "")

start = code.index("BEND_THRESHOLDS = {")
end = code.index("\n\nclass FlexDashboard:", start)
code = code[:start] + code[end + 2:]

code = code.replace(
    "        self.arm = None\n", ""
).replace(
    "        self.arm_lock = threading.Lock()\n", ""
)

start = code.index("        self.arm_enabled = False")
end = code.index("\n        ttk.Label(", start)
code = code[:start] + code[end:]

start = code.index("        arm_frame = ttk.LabelFrame(")
end = code.index("        controls = ttk.Frame(root)", start)
code = code[:start] + code[end:]

start = code.index("    def toggle_arm(self):")
end = code.index("    def reader(self):", start)
code = code[:start] + code[end:]

code = code.replace(
    '                self.update_arm_control("Thumb", values[0])\n'
    '                self.update_arm_control("Ring", values[3])\n',
    ""
)

start = code.index("        self.arm_enabled = False", code.index("    def stop(self):"))
end = code.index("        if self.ser is not None:", start)
code = code[:start] + code[end:]

code = code.replace(
    '        self.arm_status = "Arm control disabled"\n'
    '        self.arm_status_label.config(text=self.arm_status)\n',
    ""
)

path.write_text(code)
print("Removed arm controls from flex_dashboard.py")
