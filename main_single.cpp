#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <boost/process.hpp>
#include <boost/filesystem.hpp>
#include <boost/process/windows.hpp>

#define GL_SILENCE_DEPRECATION
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

string get_codec(string camera) {
    fs::path gstreamer_path_env = "C:" PATH_SEPARATOR "gstreamer" PATH_SEPARATOR "1.0" PATH_SEPARATOR "msvc_x86_64" PATH_SEPARATOR "bin" PATH_SEPARATOR "gst-launch-1.0.exe";
    string gst_launch_path;
    if (fs::exists(gstreamer_path_env)) {
        gst_launch_path = gstreamer_path_env.string();
    }
    else {
        gst_launch_path = resource_path("gstreamer" PATH_SEPARATOR "1.0" PATH_SEPARATOR "msvc_x86_64" PATH_SEPARATOR "bin" PATH_SEPARATOR "gst-launch-1.0.exe");
    }

    string command = gst_launch_path + " -v rtspsrc location=" + camera + " ! decodebin ! fakesink silent=false -m";
    string result;
    string codec = "";
    bp::ipstream pipe_stream;
    bp::child c(command, bp::std_out > pipe_stream, bp::windows::create_no_window);
    string line;
    while (pipe_stream && getline(pipe_stream, line) && !line.empty()) {
        result += line;
        if (result.find("H264") != string::npos) {
            codec = "H264";
            break;
        }
        else if (result.find("H265") != string::npos) {
            codec = "H265";
            break;
        }
    }
    c.terminate();
    if (codec.empty()) {
        cout << "H.264 or H.265 decoder not found! Skipping RTSP Url: " + camera << endl;
    }
    return codec;
}

VideoCapture set_cap(string camera, string codec) {
    string gst_str;
    if (codec == "H264") {
        gst_str = "rtspsrc location=" + camera + " latency=0 ! "
            "rtph264depay ! h264parse ! decodebin ! "
            "videoscale ! video/x-raw,width=1920,height=1080 ! "
            "videorate ! video/x-raw,framerate=24/1 !"
            "videoconvert ! appsink drop=true";
    }
    else if (codec == "H265") {
        gst_str = "rtspsrc location=" + camera + " latency=0 ! "
            "rtph265depay ! h265parse ! decodebin ! "
            "videoscale ! video/x-raw,width=1920,height=1080 ! "
            "videorate ! video/x-raw,framerate=24/1 !"
            "videoconvert ! appsink drop=true";
    }
    else {
        cout << "Unsupported video encoding for RTSP Url: " + camera << endl;
        return VideoCapture();
    }
    return VideoCapture(gst_str, CAP_GSTREAMER);
}

void process_camera(string camera, Mat& frame, Mat& rgb_frame, atomic<bool>& stop_threads) {
    string codec = get_codec(camera);
    if (codec.empty()) {
        rgb_frame = Mat::zeros(Size(1920, 1080), CV_8UC3);
        rgb_frame.setTo(Scalar(128, 128, 128));
        return;
    }

    VideoCapture cap = set_cap(camera, codec);
    cap.set(CAP_PROP_OPEN_TIMEOUT_MSEC, 1000);

    if (!cap.isOpened()) {
        cout << "Error: Could not open Capture. For this link: " << camera << endl;
        rgb_frame = Mat::zeros(Size(1920, 1080), CV_8UC3);
        rgb_frame.setTo(Scalar(128, 128, 128));
        return;
    }

    while (!stop_threads) {
        cap >> frame;
        if (frame.empty()) {
            cout << "End of video stream. Frame is empty from this link: " << camera << endl;
            rgb_frame = Mat::zeros(Size(1920, 1080), CV_8UC3);
            rgb_frame.setTo(Scalar(128, 128, 128));
            continue;
        }
        cvtColor(frame, rgb_frame, COLOR_BGR2RGB);
        cap.set(CAP_PROP_BUFFERSIZE, 2);
    }
    cap.release();
}

static void glfw_error_callback(int error, const char* description) {
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

void customBar(GLFWwindow* window, bool& show_custom_bar, int window_width, int window_height) {
    double mouse_x, mouse_y;
    glfwGetCursorPos(window, &mouse_x, &mouse_y);
    show_custom_bar = (mouse_y <= 25);

    if (show_custom_bar) {
        ImGui::SetNextWindowPos(ImVec2(window_width - 200, 0));
        ImGui::SetNextWindowSize(ImVec2(200, 30));
        ImGui::Begin("Custom Bar", &show_custom_bar, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoTitleBar);

        if (ImGui::Button("Minimize")) {
            glfwIconifyWindow(window);
        }
        ImGui::SameLine();
        if (ImGui::Button("Maximize")) {
            if (glfwGetWindowAttrib(window, GLFW_MAXIMIZED))
                glfwRestoreWindow(window);
            else
                glfwMaximizeWindow(window);
        }
        ImGui::SameLine();
        if (ImGui::Button("Close")) {
            glfwSetWindowShouldClose(window, true);
        }
        ImGui::End();
    }
}

int WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd) {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    // Set your RTSP link here
    string rtsp_link = "rtsp://admin:Cwds11354@192.168.11.7:554/Streaming/channels/1601";

    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    GLFWwindow* window = glfwCreateWindow(mode->width, mode->height, "Camera App", nullptr, nullptr);
    glfwSetWindowPos(window, 0, 0);

    if (window == nullptr)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    ImGui::StyleColorsDark();

    ImGuiStyle& style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
    io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);

    Mat frame, rgb_frame;
    vector<thread> threads;
    GLuint texture;
    bool show_custom_bar = false;
    atomic<bool> stop_threads(false);
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    glGenTextures(1, &texture);
    int window_width, window_height;
    glfwGetWindowSize(window, &window_width, &window_height);

    threads.push_back(thread(process_camera, rtsp_link, ref(frame), ref(rgb_frame), ref(stop_threads)));

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        customBar(window, show_custom_bar, window_width, window_height);

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(window_width, window_height));
        ImGui::Begin("Camera", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        if (rgb_frame.data) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb_frame.cols, rgb_frame.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_frame.data);
        }

        if (!rgb_frame.empty()) {
            ImGui::Image((void*)(intptr_t)texture, ImVec2(window_width, window_height));
        }
        ImGui::End();

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }

        glfwSwapBuffers(window);
    }

    stop_threads = true;
    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }
    glDeleteTextures(1, &texture);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
