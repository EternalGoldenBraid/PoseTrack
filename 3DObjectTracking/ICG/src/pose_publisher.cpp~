// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Manuel Stoiber, German Aerospace Center (DLR)

#include <icg/publisher.h>

namespace icg {

bool PosePublisher::set_up() const { return set_up_; }

PosePublisher::PosePublisher(const std::string &name,
                     const std::filesystem::path &metafile_path)
    : name_{name}, metafile_path_{metafile_path} {}

// By Nicklas
bool PosePublisher::UpdatePublisher(int iteration){
    std::cout << "Publisher Called" << std::endl;
    return true;
}

// Setters
PosePublisher::SetUp(){
    set_up_ = true;
}


}  // namespace icg
