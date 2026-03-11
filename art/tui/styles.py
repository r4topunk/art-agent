APP_CSS = """
Screen {
    background: #0f0f1a;
}

#header {
    dock: top;
    height: 3;
    background: #1a1a2e;
    color: #e0e0e0;
    padding: 0 2;
    content-align: center middle;
}

#main-grid {
    height: 1fr;
}

#top-row {
    height: 1fr;
}

#bottom-row {
    height: 1fr;
}

/* Top row widgets */
#training-panel {
    width: 1fr;
    min-width: 26;
    height: 100%;
    border: solid #3a3a5c;
    padding: 0 1;
    overflow: hidden;
}

#gallery-panel {
    width: 2fr;
    min-width: 44;
    height: 100%;
    border: solid #3a3a5c;
    padding: 0 1;
    overflow: hidden;
}

#evolution-panel {
    width: 1fr;
    min-width: 28;
    height: 100%;
    border: solid #3a3a5c;
    padding: 0 1;
    overflow: hidden;
}

/* Bottom row widgets */
#heartbeat {
    width: 1fr;
    min-width: 26;
    height: 100%;
    border: solid #3a3a5c;
    padding: 0 1;
    overflow: hidden;
}

#birth {
    width: 2fr;
    min-width: 44;
    height: 100%;
    border: solid #3a3a5c;
    padding: 0 1;
    overflow: hidden;
}

#timeline {
    width: 1fr;
    min-width: 28;
    height: 100%;
    border: solid #3a3a5c;
    padding: 0 1;
    overflow: hidden;
}

/* Review screen */
ReviewScreen {
    layout: vertical;
}

#review-grid {
    height: 2fr;
    border: solid #3a3a5c;
    padding: 1;
}

#review-detail {
    height: 1fr;
    border: solid #3a3a5c;
    padding: 1;
}
"""
