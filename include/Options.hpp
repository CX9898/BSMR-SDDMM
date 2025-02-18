#pragma once

#include <unistd.h>

#include <iostream>
#include <string>

class Options {
 public:
  Options(int argc, char *argv[]) {
      char param_opt;
      while ((param_opt = getopt(argc, argv, "a:b:f:k:m:")) != -1) {
          try {
              switch (param_opt) {
                  case 'a':if (optarg == nullptr) throw std::invalid_argument("Missing argument for -a");
                      alpha_ = std::stof(optarg);
                      break;
                  case 'b':if (optarg == nullptr) throw std::invalid_argument("Missing argument for -b");
                      beta_ = std::stof(optarg);
                      break;
                  case 'f':if (optarg == nullptr) throw std::invalid_argument("Missing argument for -f");
                      input_file_ = std::string(optarg);
                      break;
                  case 'k':if (optarg == nullptr) throw std::invalid_argument("Missing argument for -k");
                      K_ = std::stoi(optarg);
                      break;
                  case 'm':if (optarg == nullptr) throw std::invalid_argument("Missing argument for -m");
                      is_make_data_ = true;
                  default :std::cerr << "Unknown option: " << (char) optopt << std::endl;
                      break;
              }
          } catch (const std::invalid_argument &e) {
              std::cerr << "Invalid argument for option -" << param_opt << ": " << e.what() << std::endl;
              exit(EXIT_FAILURE);
          } catch (const std::out_of_range &e) {
              std::cerr << "Out of range error for option -" << param_opt << ": " << e.what() << std::endl;
              exit(EXIT_FAILURE);
          }
      }

      // 处理非选项参数
      if (optind < argc) {
          std::cout << "Non-option arguments: ";
          for (int i = optind; i < argc; i++) {
              std::cout << argv[i] << " ";
          }
          std::cout << std::endl;
      }
  }

 private:
  std::string input_file_;
  size_t K_;
  float alpha_;
  float beta_;

  bool is_make_data_ = false;
};