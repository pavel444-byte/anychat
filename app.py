from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Окно без рамок — выглядит как нативное приложение
        self.setWindowTitle("AnyChat App")
        self.setGeometry(100, 100, 1280, 720)

        # Веб-вьюха
        self.browser = QWebEngineView()
        self.browser.setUrl("http://172.22.190.120:8501")  # Здесь можно вставить URL своего сайта

        # Вставляем веб-вьюху в интерфейс
        central_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.browser)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
