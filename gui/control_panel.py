from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget, QLabel, QPlainTextEdit, QComboBox

from methods.lctgen import generate_scene as generate_scenario_with_lctgen

class ControlPanel(QWidget):
    def __init__(self, window):
        super().__init__()
        self.setMaximumWidth(200)

        self.window = window
        self.window.commands["update_method"] = self.update_method
        self.window.commands["update_query"] = self.update_query

        self.is_generating = False

        layout = QVBoxLayout()

        self.map_selector = QComboBox()
        self.map_selector.addItems(self.window.maps)
        self.map_selector.setCurrentIndex(-1)
        self.map_selector.currentTextChanged.connect(self.window.commands["update_map"])

        self.method_selector = QComboBox()
        self.method_selector.addItems(self.window.methods)
        self.method_selector.setCurrentIndex(0)
        self.method_selector.currentTextChanged.connect(self.window.commands["update_method"])

        self.query_input = QPlainTextEdit()
        self.query_input.setPlaceholderText("Enter query here...")
        self.query_input.setFixedHeight(300)
        self.query_input.textChanged.connect(self.window.commands["update_query"])

        self.generate_button = QLabel("Generate")
        self.generate_button.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.generate_button.setStyleSheet("background-color: #00FF00; color: #000000;")
        self.generate_button.setFixedHeight(50)
        self.generate_button.mousePressEvent = self.generate_scenario

        layout.addWidget(self.map_selector)
        layout.addWidget(self.method_selector)
        layout.addWidget(self.query_input)
        layout.addWidget(self.generate_button)

        self.setLayout(layout)
        
    def update_method(self):
        method_name = self.method_selector.currentText()
        print(f"Using {method_name} method under {self.window.env_name} environment.")
        self.window.method_name = method_name

    def update_query(self):
        query = self.query_input.toPlainText()
        print(f"Query updated: {query}")
        self.window.query = query

    def generate_scenario(self, event):
        if self.is_generating or not self.window.query or not self.window.vector_map: return

        self.is_generating = True
        print("Generating scenario...")

        self.generate_button.setStyleSheet("background-color: #FF0000; color: #FFFFFF;")
        self.generate_button.setText("Generating...")
        self.generate_button.repaint()

        # try:
        if self.window.method_name == "lctgen":
            scenario = generate_scenario_with_lctgen(self.window.query, self.window.vector_map)
            self.window.map_panel.viewer.clearInteractiveOverlay()
            for i, traj in enumerate(scenario["traj"].swapaxes(0, 1)):
                self.window.map_panel.viewer.drawTrajectory(traj, color=(255, 0, 0, 255) if i == 0 else (0, 0, 255, 255))
        # except Exception as e:
        #     print(e)
        #     print("Failed to generate scenario.")

        self.generate_button.setStyleSheet("background-color: #00FF00; color: #000000;")
        self.generate_button.setText("Generate")
        self.generate_button.repaint()

        print("Generation ends.")
        self.is_generating = False
