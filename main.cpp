#define _CRT_SECURE_NO_WARNINGS // need it to use getenv()
#include <iostream>
#include <string>
#include <cstdio>
#include <array>
#include <thread>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <boost/process.hpp>
#include <boost/filesystem.hpp>
#include <boost/process/windows.hpp>
#include <imfilebrowser.h> //https://github.com/AirGuanZ/imgui-filebrowser/tree/master  need C++ 17  


#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

#ifdef _WIN32
#define PATH_SEPARATOR "\\"
#else
#define PATH_SEPARATOR "/"
#endif

using namespace cv;
using namespace std;
namespace bp = boost::process;
namespace fs = boost::filesystem;


string resource_path(string relative_path) {
    string base_path;
    char* _MEIPASS = getenv("_MEIPASS");
    if (_MEIPASS) {
        base_path = _MEIPASS;
    }
    else {
        base_path = fs::current_path().string();
    }
    replace(relative_path.begin(), relative_path.end(), '/', PATH_SEPARATOR[0]);
    fs::path full_path = fs::path(base_path) / relative_path;
    return full_path.string();
}

pair<map<string, vector<string>>, vector<string>> read_rtsp_links(string& filename) {
    // Prompt the user for a filename using the ImGui file browser

    map<string, vector<string>> groups;
    vector<string> group_names;
    ifstream file(filename);
    if (!file) {
        cout << "Unable to open file: " << filename << endl;
        return make_pair(groups, group_names);  // Return empty map
    }
    string line;

    // Read the column titles (group names)
    getline(file, line);
    stringstream ss(line);

    string group_name;
    while (getline(ss, group_name, ',')) {  // Split line by comma character
        group_names.push_back(group_name);
        groups[group_name] = vector<string>();
    }

    // Read the RTSP links
    while (getline(file, line)) {
        stringstream ss(line);
        string link;
        int group_index = 0;
        while (getline(ss, link, ',')) {  // Split line by comma character
            if (!link.empty()) {
                groups[group_names[group_index]].push_back(link);
            }
            group_index++;
        }
    }
    // Remove duplicates from each group
    for (auto& group : groups) {
        sort(group.second.begin(), group.second.end());
        group.second.erase(unique(group.second.begin(), group.second.end()), group.second.end());
    }
    return make_pair(groups, group_names);
}

string get_codec(string camera) {
    //string gst_launch_path = "C:\\gstreamer\\1.0\\msvc_x86_64\\bin\\gst-launch-1.0.exe";

    fs::path gstreamer_path_env = "C:" PATH_SEPARATOR "gstreamer" PATH_SEPARATOR "1.0" PATH_SEPARATOR "msvc_x86_64" PATH_SEPARATOR "bin" PATH_SEPARATOR "gst-launch-1.0.exe";

    string gst_launch_path;
    if (fs::exists(gstreamer_path_env)) {
        gst_launch_path = gstreamer_path_env.string();
        cout << "Path Enviroment found!" << endl;
    }
    else {
        gst_launch_path = resource_path("gstreamer" PATH_SEPARATOR "1.0" PATH_SEPARATOR "msvc_x86_64" PATH_SEPARATOR "bin" PATH_SEPARATOR "gst-launch-1.0.exe");
        cout << "Relative Path found!" << endl;
    }

    string command = gst_launch_path + " -v rtspsrc location=" + camera + " ! decodebin ! fakesink silent=false -m";

    string result;
    string codec = "";

    // Start a new process
    bp::ipstream pipe_stream;
    bp::child c(command, bp::std_out > pipe_stream, bp::windows::create_no_window);

    // Read the output of the child process
    string line;
    while (pipe_stream && getline(pipe_stream, line) && !line.empty()) {
        result += line;

        if (result.find("H264") != string::npos) {
            cout << "H264 found" << endl;
            codec = "H264";
            break;
        }
        else if (result.find("H265") != string::npos) {
            cout << "H265 found" << endl;
            codec = "H265";
            break;
        }
    }
    // Terminate the cild process for all cases
    c.terminate();
    if (codec.empty()) {
        cout << "H.264 or H.265 decoder not found! Skipping RTSP Url: " + camera << endl;
    }
    return codec;
}

VideoCapture set_cap(string camera, string codec, int width, int height) {
    string gst_str;
    if (codec == "H264") {
        cout << "Encoding H264" << endl;
        gst_str = "rtspsrc location=" + camera + " latency=0 ! "
            "rtph264depay ! h264parse ! decodebin ! "
            "videoscale ! video/x-raw,width=" + to_string(width) + ",height=" + to_string(height) + " ! "
            "videorate !  video/x-raw,framerate=24/1 !"
            "videoconvert ! appsink drop=true";
    }
    else if (codec == "H265") {
        cout << "Encoding H265" << endl;
        gst_str = "rtspsrc location=" + camera + " latency=0 ! "
            "rtph265depay ! h265parse ! decodebin ! "
            "videoscale ! video/x-raw,width=" + to_string(width) + ",height=" + to_string(height) + " ! "
            "videorate !  video/x-raw,framerate=24/1 !"
            "videoconvert ! appsink drop=true";
    }
    else {
        cout << "Unsupported video encoding for RTSP Url: " + camera << endl;
        return VideoCapture();
    }
    return VideoCapture(gst_str, CAP_GSTREAMER);
}

VideoCapture set_cap_normal(string camera, string codec) {
    return set_cap(camera, codec, 640, 480);
}

VideoCapture set_cap_high(string camera, string codec) {
    return set_cap(camera, codec, 1920, 1080);
}

void process_cameras(string camera, Mat& frame, Mat& rgb_frame, atomic<bool>& stop_threads, bool& full_screen_mode) {
    // Using Gstreamer
    string codec = get_codec(camera);

    // Checking if camera feed can be open if not then make screen grey
    if (codec.empty()) {
        rgb_frame = Mat::zeros(Size(640, 480), CV_8UC3);
        rgb_frame.setTo(Scalar(128, 128, 128));  // Set to grey
        return;
    }
    // Uncomment this when I find a gstreamer fix and then comment fix this function over all

    VideoCapture cap;

    if (full_screen_mode) {
        cap = set_cap_high(camera, codec);
    }
    else {
        cap = set_cap_normal(camera, codec);  // Set up the video capture
    }

    cap.set(CAP_PROP_OPEN_TIMEOUT_MSEC, 1000); // Set timeout to 1 seconds

    if (!cap.isOpened()) {
        cout << "Error: Could not open Capture. For this link: " << camera << endl;
        return;
    }

    while (!stop_threads) { // Check the flag to run this loop

        cap >> frame;  // Read a frame from the pipeline

        if (frame.empty()) {
            cout << "End of video stream. Frame is empty from this link: " << camera << endl;
            rgb_frame = Mat::zeros(Size(640, 480), CV_8UC3);
            rgb_frame.setTo(Scalar(128, 128, 128));  // Set to grey
            continue;
        }

        cvtColor(frame, rgb_frame, COLOR_BGR2RGB);
        cap.set(CAP_PROP_BUFFERSIZE, 2);
       
    }
    cap.release();
  
}

void clear_frames(vector<Mat>& frames, vector<Mat>& rgb_frames) {
    for (auto& frame : frames) {
        frame.release();
    }
    for (auto& rgb_frame : rgb_frames) {
        rgb_frame.release();
    }
}

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

void generateTextures(vector<GLuint>& textures) {
    glGenTextures(textures.size(), textures.data());
}

void deleteTextures(vector<GLuint>& textures) {
    glDeleteTextures(textures.size(), textures.data());
}

void groupSelection(vector<string>& groupNames, int& selectedGroup, bool& show_group_bar) {
    if (!show_group_bar) return;
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(230, 30));  // Adjust the size as needed
    ImGui::Begin("Group Selection", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    if (ImGui::BeginCombo("Groups", groupNames[selectedGroup].c_str())) {
        for (int n = 0; n < groupNames.size(); ++n) {
            bool isSelected = (selectedGroup == n);
            if (ImGui::Selectable(groupNames[n].c_str(), isSelected)) {
                selectedGroup = n;
            }
            if (isSelected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    ImGui::End();
}

void customBar(GLFWwindow* window, bool& show_custom_bar, int window_width, int window_height)
{
    // Check if the mouse is at the top of the screen
    double mouse_x, mouse_y;
    glfwGetCursorPos(window, &mouse_x, &mouse_y);
    if (mouse_y <= 25) // adjust this value as needed
    {
        show_custom_bar = true;
    }
    else
    {
        show_custom_bar = false;
    }

    if (show_custom_bar)
    {
        // Calculate the position of the custom bar
        int pos_x = window_width - 200; // adjust this value as needed
        int pos_y = 0;

        // Set the window position and size
        ImGui::SetNextWindowPos(ImVec2(pos_x, pos_y));
        ImGui::SetNextWindowSize(ImVec2(200, 30)); // adjust these values as needed

        // Create a custom bar with ImGui
        ImGui::Begin("Custom Bar", &show_custom_bar, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoTitleBar);

        if (ImGui::Button("Minimize"))
        {
            // handle minimize
            glfwIconifyWindow(window);
        }

        ImGui::SameLine();

        if (ImGui::Button("Maximize"))
        {
            // handle maximize
            if (glfwGetWindowAttrib(window, GLFW_MAXIMIZED))
                glfwRestoreWindow(window);
            else
                glfwMaximizeWindow(window);
        }

        ImGui::SameLine();

        if (ImGui::Button("Close"))
        {
            glfwSetWindowShouldClose(window, true);
        }

        ImGui::End();
    }
}

void fileSelector(ImGui::FileBrowser& fileDialog, string& filename, map<string, vector<string>>& groups, vector<string>& groupNames, bool& show_fileSelector) {
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(150, 30));  // Adjust the size as needed
    fileDialog.SetTitle("File Browser");
    fileDialog.SetTypeFilters({ ".*" });

    ImGui::Begin("File Selector", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    // open file dialog when user clicks this button
    if (ImGui::Button("Open File Browser"))
        fileDialog.Open();

    ImGui::End();

    fileDialog.Display();

    if (fileDialog.HasSelected())
    {
        filename = fileDialog.GetSelected().string();
        pair<map<string, vector<string>>, vector<string>> result = read_rtsp_links(filename);
        groups = result.first;
        groupNames = result.second;
        cout << "Selected filename: " << filename << endl;
        fileDialog.ClearSelected();
        show_fileSelector = false;
    }
}

void fullScreen(GLFWwindow* window, int window_width, int window_height, bool& full_screen_mode, bool& show_back_button, bool& show_group_bar, bool& show_quadrants, bool& camera_starter, vector<string>& cameras, int& selectedCamera, vector<Mat>& frames, vector<Mat>& rgb_frames, vector<thread>& threads, atomic<bool>& stop_threads, vector<GLuint>& textures) {
    if (!full_screen_mode) return;

    // Displaying full screen
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(window_width, window_height));
    ImGui::Begin("Full Screen Camera", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    
    // Update texture for the full-screen mode
    glBindTexture(GL_TEXTURE_2D, textures[0]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    if (rgb_frames[0].data) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb_frames[0].cols, rgb_frames[0].rows, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_frames[0].data);
    }

    if (!rgb_frames[0].empty()) {
        ImGui::Image((void*)(intptr_t)textures[0], ImVec2(window_width, window_height));
    }
    ImGui::End();

    // Check if the mouse is at the top of the screen
    double mouse_x, mouse_y;
    glfwGetCursorPos(window, &mouse_x, &mouse_y);

    if (mouse_y <= 25) // adjust this value as needed
    {
        show_back_button = true;
    }
    else
    {
        show_back_button = false;
    }

    if (ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Escape)))
    {
        // handle ending the full screeen mode
        full_screen_mode = false;
        show_back_button = false;
        stop_threads = true;
        show_group_bar = true;
        show_quadrants = true;
        camera_starter = true;
        // Ensure textures and threads are cleaned up properly
        for (auto& th : threads) {
            if (th.joinable()) {
                th.join();
            }
        }
        threads.clear();
        clear_frames(frames, rgb_frames);
        stop_threads = false;  // Reset stop_threads for further use
    }

    if (show_back_button)
    {
        // Set the window position and size
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(200, 30)); // adjust these values as needed

        // Create a custom bar with ImGui
        ImGui::Begin("Back Button", &show_back_button, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoTitleBar);

        if (ImGui::Button("Back"))
        {
            // handle ending the full screeen mode
            full_screen_mode = false;
            show_back_button = false;
            stop_threads = true;
            show_group_bar = true;
            show_quadrants = true;
            camera_starter = true;
            // Ensure textures and threads are cleaned up properly
            for (auto& th : threads) {
                if (th.joinable()) {
                    th.join();
                }
            }
            threads.clear();
            clear_frames(frames, rgb_frames);
            stop_threads = false;  // Reset stop_threads for further use
        }

        ImGui::End();
    }
}

// Main code
int WinMain(
               HINSTANCE hInstance,
               HINSTANCE hPrevInstance,
               LPSTR     lpCmdLine,
               int       nShowCmd
)
{
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;
    // Parameter that neeed to be defined
    string filename;
    map<string, vector<string>> groups;
    vector<string> groupNames;
    string selectedGroupName;
    vector<string> cameras;
    vector<Mat> frames(6);
    vector<Mat> rgb_frames(6);
    vector<thread> threads;
    vector<GLuint> textures(6);
    int selectedGroup = 0;
    int selectedCamera = -1;
    bool show_custom_bar = false;
    bool show_fileSelector = true;
    bool camera_starter = true;
    bool full_screen_mode = false;
    bool show_back_button = false;
    bool show_group_bar = true;
    bool show_single_camera = true;
    bool show_quadrants = true;
    atomic<bool> stop_threads(false); // Flag for stopping all camera feed and starting new one

    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    // Create window with graphics context

    // Get the primary monitor
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();

    // Get the resolution of the monitor
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);

    // Create window with the monitor's resolution
    GLFWwindow* window = glfwCreateWindow(mode->width, mode->height, "Camera App", nullptr, nullptr);

    // Set the window to fullscreen fix this on monitor without borders
    //glfwSetWindowMonitor(window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
    // With borders at bottom
    glfwSetWindowPos(window, 0, 0);


    if (window == nullptr)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    //io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;         // Enable Docking
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;       // Enable Multi-Viewport / Platform Windows
    //io.ConfigViewportsNoAutoMerge = true;
    //io.ConfigViewportsNoTaskBarIcon = true;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
    ImGuiStyle& style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
    io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
    // Our state
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    // Generate textures outside the main loop
    generateTextures(textures);

    // Getting the window width and height for setting up other windows
    int window_width, window_height;
    glfwGetWindowSize(window, &window_width, &window_height);

    // create a file browser instance
    ImGui::FileBrowser fileDialog;

    // Calculate the quadrant size
    int quadrant_width = window_width / 3;
    int quadrant_height = (window_height - 30) / 2;  // Since you want to display 6 cameras, divide height by 2

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (show_fileSelector) {
            // Calling custome bar
            customBar(window, show_custom_bar, window_width, window_height);
            // Selecting file is set to false after is use
            fileSelector(fileDialog, filename, groups, groupNames, show_fileSelector);
        }

        else
        {
            // Calling custome bar
            customBar(window, show_custom_bar, window_width, window_height);

            // Creating a holder value for old groups
            int oldSelectedGroup = selectedGroup;


            // Create group selection tap window on the top left corner
            groupSelection(groupNames, selectedGroup, show_group_bar);

            if (oldSelectedGroup != selectedGroup) {
                // Signal the threads to stop
                stop_threads = true;

                // Stop the currently running threads
                for (auto& t : threads) {
                    if (t.joinable()) {
                        t.join();
                    }
                }
                threads.clear();

                // Clear the frames
                clear_frames(frames, rgb_frames);
                // Reset the flag
                stop_threads = false;
                // Set camera_starter to true to start new threads
                camera_starter = true;
            }

            // Get the selected group. Keep outside for updating the if(camera_starter)
            selectedGroupName = groupNames[selectedGroup];


            // Need to call this everytime I exit out of full screen as well hence the if (show_quadrants) is need it
            // Starting the cameras One time in the loop
            if (show_quadrants) {
                if (camera_starter) {
                    cameras = groups[selectedGroupName];
                    if (!cameras.empty()) {
                        for (int i = 0; i < 6; ++i) {
                            threads.push_back(thread(process_cameras, cameras[i], ref(frames[i]), ref(rgb_frames[i]), ref(stop_threads), ref(full_screen_mode)));
                        }
                    }
                    // These prints are for testing 
                    cout << selectedGroupName << endl;
                    cout << cameras[0] << endl;
                    camera_starter = false;
                }
            }

            if (full_screen_mode) {
                if (show_single_camera) {
                    // Stop all threads and clear frames
                    stop_threads = true;
                    for (auto& th : threads) {
                        if (th.joinable()) {
                            th.join();
                        }
                    }
                    threads.clear();
                    clear_frames(frames, rgb_frames);

                    // Start the camera thread for the selected camera
                    stop_threads = false;
                    threads.push_back(thread(process_cameras, cameras[selectedCamera], ref(frames[0]), ref(rgb_frames[0]), ref(stop_threads), ref(full_screen_mode)));
                    show_single_camera = false; // Ensure this only happens once
                }
                // Show the selected camera in full screen
                fullScreen(window, window_width, window_height, full_screen_mode, show_back_button, show_group_bar, show_quadrants, camera_starter, cameras, selectedCamera, frames, rgb_frames, threads, stop_threads, textures);
            }

            if (show_quadrants) {
                show_single_camera = true;
                // Create an ImGui window for each camera
                for (int i = 0; i < 6; ++i) {

                    // Calculate the position of the current quadrant
                    int pos_x = (i % 3) * quadrant_width;
                    int pos_y = 30 + (i / 3) * quadrant_height;

                    // Set the window position and size. This is for multiple small screen
                    ImGui::SetNextWindowPos(ImVec2(pos_x, pos_y));
                    ImGui::SetNextWindowSize(ImVec2(quadrant_width, quadrant_height));

                    string windowTitle = "Group: " + selectedGroupName + ", Camera " + to_string(i);
                    ImGui::Begin(windowTitle.c_str(), NULL, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollWithMouse);
                    glBindTexture(GL_TEXTURE_2D, textures[i]);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
                    if (rgb_frames[i].data) {
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb_frames[i].cols, rgb_frames[i].rows, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_frames[i].data);
                    }

                    // Display the texture in the ImGui window
                    ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(textures[i])), ImVec2(rgb_frames[i].cols, rgb_frames[i].rows));

                    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
                        // Set flag to initiate full screen mode handling
                        selectedCamera = i;
                        full_screen_mode = true;
                        show_back_button = true;
                        show_group_bar = false;
                        show_quadrants = false;

                    }
                    ImGui::End();
                }
            }

        }

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }

        glfwSwapBuffers(window);
    }

    // Delete textures after the main loop
    deleteTextures(textures);

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
