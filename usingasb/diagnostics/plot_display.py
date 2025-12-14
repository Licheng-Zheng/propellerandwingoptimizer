from complete_diagnostic_helpers import plot_cst_display

if __name__ == "__main__":
    INPUT_DIR = "usingasb/diagnostics/Modal_Display_Info"
    OUTPUT_DIR = "usingasb/diagnostics/Modal_Display"

    plot_cst_display.process_all_cst_displays(INPUT_DIR, OUTPUT_DIR)