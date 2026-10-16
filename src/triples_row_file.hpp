#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include "triples_sector_checkpoint.hpp"
#if defined(__linux__)
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace localized_cc::detail {
// Rank-private binary64 rows: temporary by default, durable with an explicit
// name and identity. Durable files are retained, never unlinked by destruction.
// The caller assembles one row at a time; reads use positional I/O and are
// safe across OpenMP readers after finish_writes(). No mmap/RSS assumption.
class TriplesRowFile {
 public:
  TriplesRowFile(const std::string& directory, std::size_t rows, std::size_t columns,
                const std::string& persistent_name={}, const std::string& identity={})
      : rows_(rows), columns_(columns) {
#if defined(__linux__)
    if (columns && rows > std::numeric_limits<std::size_t>::max()/columns/sizeof(double))
      throw std::overflow_error("triples row file size overflow");
    const auto bytes=rows*columns*sizeof(double);
    if (bytes>static_cast<std::size_t>(std::numeric_limits<off_t>::max()))
      throw std::overflow_error("triples row file offset overflow");
    if (!std::filesystem::is_directory(directory))
      throw std::invalid_argument("triples scratch directory must already exist");
    if(!persistent_name.empty()) {
      if(rows>60000) throw std::invalid_argument("moment row manifest exceeds supported size");
      persistent_=true;
      path_=(std::filesystem::path(directory)/persistent_name).string();
      identity_=identity+":"+std::to_string(rows)+":"+std::to_string(columns);
      if(identity.empty()) throw std::invalid_argument("moment checkpoint requires identity");
      const bool exists=std::filesystem::exists(path_);
      fd_=::open(path_.c_str(),O_RDWR|(exists ? 0 : O_CREAT|O_EXCL),0600);
      if(fd_<0) throw std::runtime_error("cannot open moment checkpoint");
      try {
        if(::flock(fd_,LOCK_EX|LOCK_NB)!=0)
          throw std::runtime_error("moment checkpoint already in use; refusing concurrent restart");
        if(exists && !std::filesystem::exists(path_+".rows"))
          throw std::runtime_error("orphan moment file without manifest; use a new checkpoint directory");
        if(exists && std::filesystem::file_size(path_)!=bytes)
          throw std::runtime_error("moment checkpoint data size mismatch");
        const int error=bytes ? ::posix_fallocate(fd_,0,static_cast<off_t>(bytes)) : 0;
        if(error) throw std::runtime_error("moment checkpoint disk reservation failed: "+std::string(std::strerror(error))+"; requested bytes="+std::to_string(bytes));
        record_=read_sector_checkpoint(path_+".rows",identity_,1+2*rows);
        if(record_[0]<0 || record_[0]>static_cast<double>(rows) || std::floor(record_[0])!=record_[0])
          throw std::runtime_error("invalid moment checkpoint row count");
        completed_=static_cast<std::size_t>(record_[0]);
        std::vector<double> row(columns);
        for(std::size_t r=0;r<completed_;++r) {
          transfer(false,r*columns,row.data(),columns);
          const auto hash=row_hash(row);
          if(record_[1+2*r]!=static_cast<double>(hash>>32) ||
             record_[2+2*r]!=static_cast<double>(hash&0xffffffffULL))
            throw std::runtime_error("moment checkpoint row checksum mismatch");
        }
        if(!exists) {
          if(::fsync(fd_)!=0) throw std::runtime_error("moment checkpoint initial fsync failed");
          write_sector_checkpoint(path_+".rows",identity_,record_);
        }
      } catch(...) { cleanup(); throw; }
      return;
    }
    std::string pattern=(std::filesystem::path(directory)/"triples-rows-XXXXXX").string();
    std::vector<char> name(pattern.begin(),pattern.end()); name.push_back('\0');
    fd_=::mkstemp(name.data());
    if(fd_<0) throw std::runtime_error("cannot create exclusive triples scratch file: "+std::string(std::strerror(errno)));
    path_=name.data();
    // Reserve real disk blocks. Quota/space failure occurs before computation,
    // not hours later. Unsupported allocation is an error, not sparse fallback.
    const int error=bytes ? ::posix_fallocate(fd_,0,static_cast<off_t>(bytes)) : 0;
    if(error) { cleanup(); throw std::runtime_error("triples scratch reservation failed: "+std::string(std::strerror(error))); }
#else
    (void)directory;
    throw std::runtime_error("disk-backed triples requires Linux positional I/O");
#endif
  }
  ~TriplesRowFile() { cleanup(); }
  TriplesRowFile(const TriplesRowFile&)=delete;
  TriplesRowFile& operator=(const TriplesRowFile&)=delete;
  std::size_t completed_rows() const { return completed_; }
  void commit_row(std::size_t row) {
    if(!persistent_) return;
    if(row>=rows_ || row>completed_) throw std::logic_error("non-prefix moment checkpoint commit");
    if(cached_row_!=row) {
      flush(); cache_.assign(columns_,0.0); cached_row_=row;
    }
    flush();
#if defined(__linux__)
    if(::fsync(fd_)!=0) throw std::runtime_error("moment checkpoint data fsync failed");
#endif
    const auto hash=row_hash(cache_);
    record_[1+2*row]=static_cast<double>(hash>>32);
    record_[2+2*row]=static_cast<double>(hash&0xffffffffULL);
    completed_=std::max(completed_,row+1); record_[0]=static_cast<double>(completed_);
    write_sector_checkpoint(path_+".rows",identity_,record_);
  }
  void set(std::size_t row,std::size_t column,double value) {
    if(row>=rows_ || column>=columns_ || sealed_) throw std::out_of_range("invalid triples scratch write");
    if(row!=cached_row_) {
      flush();
      if(cached_row_!=invalid && row<cached_row_)
        throw std::logic_error("triples scratch producer is not row-monotone");
      cache_.assign(columns_,0.0); cached_row_=row;
    }
    cache_[column]=value;
  }
  void finish_writes() {
    if(sealed_) return;
    flush(); sealed_=true;
    std::vector<double>().swap(cache_);
  }
  void read(std::size_t first,std::span<double> output) const {
    if(!sealed_ || first>rows_ || (columns_==0 ? !output.empty() : output.size()/columns_>rows_-first || output.size()%columns_!=0))
      throw std::out_of_range("invalid triples scratch read");
    transfer(false,first*columns_,output.data(),output.size());
  }
 private:
  static constexpr auto invalid=std::numeric_limits<std::size_t>::max();
  std::size_t rows_,columns_,cached_row_{invalid};
  bool sealed_{false};
  bool persistent_{false};
  std::size_t completed_{};
  std::string identity_;
  std::vector<double> record_;
  static std::uint64_t row_hash(const std::vector<double>& row) {
    if(row.empty()) return sector_checksum("");
    return sector_checksum(std::string(reinterpret_cast<const char*>(row.data()),row.size()*sizeof(double)));
  }
  std::vector<double> cache_;
  std::string path_;
  int fd_{-1};
  void flush() { if(!cache_.empty()) transfer(true,cached_row_*columns_,cache_.data(),cache_.size()); }
  void transfer(bool write,std::size_t first,double* data,std::size_t count) const {
#if defined(__linux__)
    auto* p=reinterpret_cast<char*>(data);
    std::size_t bytes=count*sizeof(double),offset=first*sizeof(double);
    while(bytes) {
      const auto chunk=std::min(bytes,std::size_t{1}<<30);
      const auto n=write ? ::pwrite(fd_,p,chunk,static_cast<off_t>(offset))
                         : ::pread(fd_,p,chunk,static_cast<off_t>(offset));
      if(n<0 && errno==EINTR) continue;
      if(n<=0) throw std::runtime_error("triples scratch I/O failed or truncated file");
      p+=n; offset+=static_cast<std::size_t>(n); bytes-=static_cast<std::size_t>(n);
    }
#else
    (void)write; (void)first; (void)data; (void)count;
    throw std::runtime_error("disk-backed triples requires Linux");
#endif
  }
  void cleanup() noexcept {
#if defined(__linux__)
    if(fd_>=0) { ::close(fd_); fd_=-1; }
    if(!path_.empty() && !persistent_) { ::unlink(path_.c_str()); path_.clear(); }
#endif
  }
};
} // namespace localized_cc::detail
