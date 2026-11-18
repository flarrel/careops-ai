#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

using nlohmann::json;

std::size_t receive(char* data, std::size_t size, std::size_t count, void* destination) {
    auto& body = *static_cast<std::string*>(destination);
    const auto bytes = size * count;
    if (body.size() + bytes > 1024 * 1024) {
        return 0;
    }
    body.append(data, bytes);
    return bytes;
}

int main() {
    try {
        std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(), curl_easy_cleanup);
        if (!curl) {
            throw std::runtime_error("Could not initialize HTTP client");
        }
        const char* configured = std::getenv("CAREOPS_API_URL");
        std::string endpoint = configured ? configured : "http://127.0.0.1:8000";
        if (!endpoint.empty() && endpoint.back() == '/') {
            endpoint.pop_back();
        }
        endpoint += "/runs?limit=10";

        std::string body;
        curl_easy_setopt(curl.get(), CURLOPT_URL, endpoint.c_str());
        curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT, 5L);
        curl_easy_setopt(curl.get(), CURLOPT_NOPROXY, "*");
        curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, receive);
        curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &body);
        const auto result = curl_easy_perform(curl.get());
        long status = 0;
        curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &status);
        if (result != CURLE_OK || status != 200) {
            throw std::runtime_error("CareOps API unavailable");
        }

        const auto rows = json::parse(body);
        if (!rows.is_array()) {
            throw std::runtime_error("Expected an array of run summaries");
        }
        std::map<std::string, int> outcomes;
        for (const auto& row : rows) {
            ++outcomes[row.at("status").get<std::string>()];
        }
        std::cout << json{{"runs", rows.size()}, {"outcomes", outcomes}}.dump() << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Run summary unavailable. " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
