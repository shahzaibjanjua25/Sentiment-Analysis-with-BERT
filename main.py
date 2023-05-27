import sys
from PyQt5.QtWidgets import QApplication
from gui import SentimentAnalysisWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SentimentAnalysisWindow()
    window.show()
    sys.exit(app.exec_())
