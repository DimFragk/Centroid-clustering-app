import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from dataclasses import dataclass, field
from typing import Optional, Callable, Literal
from streamlit_option_menu import option_menu

# import Global_system_functions as mf
from . import streamlit_functions as Ui
from .general_functions import get_last_name_of


# import Read_Excel as rxl


@dataclass(slots=True, frozen=True)
class PageTab:
    tab_name: str
    tab_page: Callable
    tab_icon: Optional[str] = None
    tab_setting: Optional[Callable] | Optional[True] = field(default=None)    # Optional[Callable] = None


def create_render_container_pages(all_tabs, tab_type, all_tab_icons=None):
    print("gdss_app_line_60")
    print(f"tab_type: {tab_type}")

    if tab_type == "Options menu":
        with st.sidebar:
            st.divider()
            st.title("System menu")
            render = option_menu(
                menu_title=None,  # "System menu",
                options=all_tabs,
                orientation="vertical",
                icons=all_tab_icons  # [all_tab_icons[icon] for icon in all_tab_icons]
            )
        containers_dict = {tab_name: st.container() for tab_name in all_tabs}

    elif tab_type == "Radio tabs":

        render = [st.radio(
            label="Select which results to render",
            options=all_tabs,
            index=0, horizontal=True, label_visibility="collapsed"
        )]

        containers_dict = {tab_name: st.container() for tab_name in all_tabs}

    elif tab_type == "Multiselect tabs":

        with st.sidebar:
            render = st.multiselect(
                label="Select which results to render",
                options=all_tabs,
                default=all_tabs,
                key="Tabs Render Menu"
            )

        containers_dict = {tab_name: st_tab for tab_name, st_tab in zip(all_tabs, (st.tabs(all_tabs)))}

    else:
        with st.sidebar:
            render = [st.selectbox(
                label="Select result page",
                options=all_tabs,
                index=0
            )]

        containers_dict = {tab_name: st.container() for tab_name in all_tabs}

    return render, containers_dict


def render_tab_pages_dict(tabs_page_dict, all_tab_icons=None):
    all_tabs = list(tabs_page_dict.keys())
    render, containers_dict = create_render_container_pages(
        all_tabs, tab_type=st.session_state.tab_type, all_tab_icons=all_tab_icons
    )

    for tab_name in tabs_page_dict:
        if tab_name in render if isinstance(render, list) else tab_name == render:
            with containers_dict[tab_name]:
                print("GDSS_App_line_102")
                print(render)
                print(tab_name)

                if not isinstance(tabs_page_dict[tab_name], tuple):
                    tabs_page_dict[tab_name]()
                    continue

                for page in tabs_page_dict[tab_name]:
                    page()


def render_tab_pages(res_obj, tabs_page_data: list[PageTab], all_tab_icons=None):
    if isinstance(tabs_page_data, dict):
        render_tab_pages_dict(tabs_page_data, all_tab_icons=all_tab_icons)
        return

    tab_names = [pt.tab_name for pt in tabs_page_data]
    all_tab_icons = [pt.tab_icon for pt in tabs_page_data] if all_tab_icons is None else all_tab_icons
    render, containers_dict = create_render_container_pages(
        all_tabs=tab_names, tab_type=st.session_state.tab_type, all_tab_icons=all_tab_icons
    )

    for i, tab_name in enumerate(tab_names):
        if tab_name in render if isinstance(render, list) else tab_name == render:
            with containers_dict[tab_name]:
                if tabs_page_data[i].tab_setting is not None:
                    tabs_page_data[i].tab_setting(res_obj)
                tabs_page_data[i].tab_page(res_obj)


def selection_of_input_system_runs(
        from_file_to_input_obj: Callable, input_obj_set_up: Callable, random_data_gen: Callable = None,
        r_xl_type: str | Literal["wb", "bytes_io", "pandas"] = "wb", submit_set_up_needed=True
):

    # method_input = None
    with st.sidebar:
        side_bar_cont = st.container()

    side_bar_cont.title("Input data")

    rand_opt = []
    if random_data_gen is not None:
        rand_opt = ["Random cluster data demo"]

    options_list = rand_opt + ["Upload a file"] + list(st.session_state.method_input.keys())
    selection = side_bar_cont.selectbox(
        label="Select input",
        options=options_list,
        index=len(options_list) - 1
        # , on_change=change_input_data_select_box
    )

    st.session_state.run_options_slt = selection

    if selection == "Upload a file":
        with side_bar_cont:
            uploaded_file = st.file_uploader(label="Upload '.xlsx' or '.csv' file", label_visibility="collapsed")
        if not uploaded_file:
            return
        f_type_str = get_last_name_of(uploaded_file.name)
        if f_type_str == "xlsx":
            if r_xl_type == "wb":
                bytes_io_file, f_name = Ui.return_file_in_bytes_io_from(uploaded_file)
                input_data = Ui.load_workbook_from_bytes(bytes_io_file)
                """
                input_data, f_name = Ui.return_excel_workbook_with_st_file_uploader(
                    label="Input data excel workbook"
                )
                """
            elif r_xl_type == "bytes_io":
                input_data, f_name = Ui.return_file_in_bytes_io_from(uploaded_file)
            elif r_xl_type == "pandas":
                # input_data = pd.read_excel(BytesIO(uploaded_file.getvalue()))
                input_data = pd.read_excel(uploaded_file)
                f_name = uploaded_file.name
            else:
                input_data = None
                f_name = None
        elif f_type_str == "csv":
            input_data = pd.read_csv(uploaded_file)
            f_name = uploaded_file.name
        else:
            input_data = None
            f_name = None

        if input_data is not None:
            # method_input = Utgp.get_group_uta_t1_obj_from_wb(my_workbook, f_name)
            setting_up_obj = from_file_to_input_obj(input_data, f_name)
            if setting_up_obj is not None:
                st.session_state.method_input["Setting up..."] = setting_up_obj
                st.rerun()

    elif selection == "Random cluster data demo":
        # method_input = pd.DataFrame(mf.return_list_of_skewed_dists_bounded(num_of_lists=100, list_size=5))
        res = random_data_gen()
        if res is not None:
            st.session_state.method_input["Setting up..."] = res
            st.rerun()

    if selection == "Setting up...":
        if submit_set_up_needed:
            submit_set_up(rtn_method_obj_func=input_obj_set_up, method_input=st.session_state.method_input[selection])
        else:
            input_obj_set_up(st.session_state.method_input[selection])


def submit_set_up(rtn_method_obj_func: Callable, method_input):
    with st.spinner("running..."):
        res_obj, name_of_run = rtn_method_obj_func(method_input)

    if res_obj is None:
        return

    st.session_state.res_obj_dict[name_of_run] = res_obj
    if "Setting up..." in st.session_state.method_input.keys():
        del st.session_state.method_input["Setting up..."]
    st.session_state.method_input[name_of_run] = method_input
    st.session_state.st_counter += 1


def global_non_system_settings(page_settings_names: list, **other_st_stg):

    st.title("App settings")
    r_slt = st.multiselect(
        label="Render menu",
        options=["Settings placement", "Color theme", "Other"],
        default=None
    )

    if "Settings placement" in r_slt:
        with st.expander(label="Settings placement "):
            def radio_place_f(name, index=1):
                opt_dict = {
                    True: "Sidebar",
                    False: "Tab",
                    None: "Settings only"
                }
                return st.radio(
                    label=f"Show {name} settings in the sidebar",
                    options=opt_dict.keys(), index=index, horizontal=True, format_func=opt_dict.get
                )

            st.session_state.settings_placement = {name: radio_place_f(name, index=1) for name in page_settings_names}
    else:
        st.session_state.settings_placement = {name: False for name in page_settings_names}

    if "Color theme" in r_slt:
        with st.expander(label="Color theme"):
            Ui.color_theme_main()

    if "Other" in r_slt:
        with st.expander(label="Other settings"):
            st.session_state.show_info = st.toggle(label="Show helpful info", value=True)

            st.session_state.tab_type = st.radio(
                label="Create result tabs",
                options=["Options menu", "Page", "Radio tabs", "Multiselect tabs"],
            )
            for key, item_fn in other_st_stg.items():
                st.session_state.other_stg[key] = item_fn()


def main(
        page_tabs_data: list[PageTab] | dict[str, Callable | tuple[Callable]],
        from_file_to_input_obj: Callable,
        input_obj_set_up: Callable,
        random_data_gen: Callable = None,
        home_page: Callable = None,
        r_xl_type="wb",
        submit_set_up_needed=True
):
    print("Rerun:\nStart of main()...")

    if "settings_placement" not in st.session_state:
        st.session_state.settings_placement = {}

    if "show_info" not in st.session_state:
        st.session_state.show_info = True

    if "method_input" not in st.session_state:
        st.session_state.method_input = {}

    if "run_options_slt" not in st.session_state:
        st.session_state.run_options_slt = None

    if "st_counter" not in st.session_state:
        st.session_state.st_counter = 0

    if "res_obj_dict" not in st.session_state:
        st.session_state.res_obj_dict = {}

    if "tab_type" not in st.session_state:
        st.session_state.tab_type = "Options menu"

    if "other_stg" not in st.session_state:
        st.session_state.other_stg = {}

    # method_input = None
    # st.session_state.run_options_slt = None

    page_settings_names = []
    for page_obj in page_tabs_data:
        if page_obj.tab_setting is not None:
            page_settings_names += [page_obj.tab_name]

    with st.sidebar:
        global_non_system_settings(page_settings_names)
        st.divider()

    selection_of_input_system_runs(
        from_file_to_input_obj=from_file_to_input_obj,
        input_obj_set_up=input_obj_set_up,
        random_data_gen=random_data_gen,
        r_xl_type=r_xl_type,
        submit_set_up_needed=submit_set_up_needed
    )

    if st.session_state.run_options_slt not in st.session_state.res_obj_dict.keys():
        if home_page is None:
            st.divider()
            st.caption("Method is not set up")
        elif st.session_state.run_options_slt == "Upload a file":
            home_page()
        return

    selection: str | None = st.session_state.run_options_slt

    res_obj = st.session_state.res_obj_dict[selection]

    render_tab_pages(res_obj, page_tabs_data)
