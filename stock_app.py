from command_line import cmd_handler


def launch_cmd():
    cmd_pilot = cmd_handler()
    cmd_pilot.set_ticker()
    cmd_pilot.set_hist_start()
    cmd_pilot.set_hist_end()
    cmd_pilot.set_indicators()
    cmd_pilot.set_indicator_window()
    # cmd_pilot.set_model()
    print(cmd_pilot.get_input_summary())
    hist_plot, rsi_plot, msg = cmd_pilot.hist_plotter()
    if msg is None:
        hist_plot.show()
    else:
        print(f"!!!!!!!!!!!!    ERROR    !!!!!!!!!!!!\n{msg}")


def user_choice():
    inp = input("Enter your choice : ")
    done = False
    while not done:
        try:
            inp = int(inp)
        except ValueError:
            print("Input is not a valid numeric character")
        else:
            if inp in [1, 2]:
                return inp
            else:
                print(
                    "Entered number is not in the choice list (Allowed : [1, 2])")
                inp = input("Enter your choice : ")


if __name__ == "__main__":
    print("---------------     STOCK HISTORY AND FORECASTING APPLICATION     -----------------")
    print("Following modes are available for the app")
    print("\t 1. Web Appication - GUI")
    print("\t 2. Command line")
    choice = user_choice()
    if choice == 1:
        exec(open("dash_app.py").read())
    elif choice == 2:
        launch_cmd()
