! Dominant closed-shell-reference CCSD intermediate kernels.
!
! Arrays use the row-major offsets owned by localized_cc::Tensor.  MPI work
! ownership is supplied by C++; this file contains no MPI calls.
module lccsdt_legacy_intermediate_kernels
  use, intrinsic :: iso_c_binding, only: c_double, c_int, c_int64_t
  implicit none
  private
  public :: lccsdt_legacy_wabef, lccsdt_legacy_wabef_block
  public :: lccsdt_legacy_wabef_block_pair_scalars

contains

  pure integer(c_int64_t) function matrix_offset(row, column, columns) result(offset)
    integer(c_int64_t), value :: row, column, columns
    offset = 1_c_int64_t + row * columns + column
  end function matrix_offset

  pure integer(c_int64_t) function tensor4_offset(i, j, k, l, dim_j, dim_k, dim_l) result(offset)
    integer(c_int64_t), value :: i, j, k, l, dim_j, dim_k, dim_l
    offset = 1_c_int64_t + (((i * dim_j + j) * dim_k + k) * dim_l + l)
  end function tensor4_offset

  subroutine lccsdt_legacy_wabef(rank, ranks, occupied, virtuals, eri, t1, tau, wabef, status) &
      bind(C, name="lccsdt_legacy_wabef")
    integer(c_int64_t), value, intent(in) :: rank, ranks, occupied, virtuals
    real(c_double), intent(in) :: eri(*), t1(*), tau(*)
    ! In distributed mode only rank-owned blocks are written; C++ initializes
    ! all other entries to zero before the collective sum.
    real(c_double), intent(inout) :: wabef(*)
    integer(c_int), intent(out) :: status
    integer(c_int64_t) :: a, b, e, q, m, n, orbitals, task
    real(c_double) :: value

    status = 0_c_int
    if (rank < 0_c_int64_t .or. ranks <= 0_c_int64_t .or. rank >= ranks .or. &
        occupied <= 0_c_int64_t .or. virtuals <= 0_c_int64_t) then
      status = 1_c_int
      return
    end if
    orbitals = occupied + virtuals

    !$omp parallel do default(none) schedule(static) &
    !$omp shared(rank, ranks, occupied, virtuals, orbitals, eri, t1, tau, wabef) &
    !$omp private(a, b, e, q, m, n, value)
    do task = rank, virtuals * virtuals - 1_c_int64_t, ranks
      a = task / virtuals
      b = modulo(task, virtuals)
      do e = 0_c_int64_t, virtuals - 1_c_int64_t
        do q = 0_c_int64_t, virtuals - 1_c_int64_t
          value = eri(tensor4_offset(occupied + a, occupied + b, &
                                     occupied + e, occupied + q, &
                                     orbitals, orbitals, orbitals))
          do m = 0_c_int64_t, occupied - 1_c_int64_t
            value = value - &
                t1(matrix_offset(m, b, virtuals)) * &
                eri(tensor4_offset(occupied + a, m, occupied + e, occupied + q, &
                                   orbitals, orbitals, orbitals)) + &
                t1(matrix_offset(m, a, virtuals)) * &
                eri(tensor4_offset(occupied + b, m, occupied + e, occupied + q, &
                                   orbitals, orbitals, orbitals))
          end do
          do m = 0_c_int64_t, occupied - 1_c_int64_t
            do n = 0_c_int64_t, occupied - 1_c_int64_t
              value = value + 0.25_c_double * &
                  tau(tensor4_offset(m, n, a, b, occupied, virtuals, virtuals)) * &
                  eri(tensor4_offset(m, n, occupied + e, occupied + q, &
                                     orbitals, orbitals, orbitals))
            end do
          end do
          wabef(tensor4_offset(a, b, e, q, virtuals, virtuals, virtuals)) = value
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine lccsdt_legacy_wabef

  subroutine lccsdt_legacy_wabef_block(occupied, virtuals, a, b, eri, t1, tau, wabef, status) &
      bind(C, name="lccsdt_legacy_wabef_block")
    integer(c_int64_t), value, intent(in) :: occupied, virtuals, a, b
    real(c_double), intent(in) :: eri(*), t1(*), tau(*)
    real(c_double), intent(out) :: wabef(*)
    integer(c_int), intent(out) :: status
    integer(c_int64_t) :: e, q, m, n, orbitals, task
    real(c_double) :: value

    status = 0_c_int
    if (occupied <= 0_c_int64_t .or. virtuals <= 0_c_int64_t .or. &
        a < 0_c_int64_t .or. a >= virtuals .or. &
        b < 0_c_int64_t .or. b >= virtuals) then
      status = 1_c_int
      return
    end if
    orbitals = occupied + virtuals

    !$omp parallel do default(none) schedule(static) &
    !$omp shared(occupied, virtuals, a, b, orbitals, eri, t1, tau, wabef) &
    !$omp private(e, q, m, n, value)
    do task = 0_c_int64_t, virtuals * virtuals - 1_c_int64_t
      e = task / virtuals
      q = modulo(task, virtuals)
      value = eri(tensor4_offset(occupied + a, occupied + b, &
                                 occupied + e, occupied + q, &
                                 orbitals, orbitals, orbitals))
      do m = 0_c_int64_t, occupied - 1_c_int64_t
        value = value - &
            t1(matrix_offset(m, b, virtuals)) * &
            eri(tensor4_offset(occupied + a, m, occupied + e, occupied + q, &
                               orbitals, orbitals, orbitals)) + &
            t1(matrix_offset(m, a, virtuals)) * &
            eri(tensor4_offset(occupied + b, m, occupied + e, occupied + q, &
                               orbitals, orbitals, orbitals))
      end do
      do m = 0_c_int64_t, occupied - 1_c_int64_t
        do n = 0_c_int64_t, occupied - 1_c_int64_t
          value = value + 0.25_c_double * &
              tau(tensor4_offset(m, n, a, b, occupied, virtuals, virtuals)) * &
              eri(tensor4_offset(m, n, occupied + e, occupied + q, &
                                 orbitals, orbitals, orbitals))
        end do
      end do
      wabef(matrix_offset(e, q, virtuals)) = value
    end do
    !$omp end parallel do
  end subroutine lccsdt_legacy_wabef_block

  subroutine lccsdt_legacy_wabef_block_pair_scalars(occupied, virtuals, a, b, eri, t1, tau_ab, wabef, status) &
      bind(C, name="lccsdt_legacy_wabef_block_pair_scalars")
    integer(c_int64_t), value, intent(in) :: occupied, virtuals, a, b
    real(c_double), intent(in) :: eri(*), t1(*), tau_ab(*)
    real(c_double), intent(out) :: wabef(*)
    integer(c_int), intent(out) :: status
    integer(c_int64_t) :: e, q, m, n, orbitals, task
    real(c_double) :: value

    status = 0_c_int
    if (occupied <= 0_c_int64_t .or. virtuals <= 0_c_int64_t .or. &
        a < 0_c_int64_t .or. a >= virtuals .or. &
        b < 0_c_int64_t .or. b >= virtuals) then
      status = 1_c_int
      return
    end if
    orbitals = occupied + virtuals

    !$omp parallel do default(none) schedule(static) &
    !$omp shared(occupied, virtuals, a, b, orbitals, eri, t1, tau_ab, wabef) &
    !$omp private(e, q, m, n, value)
    do task = 0_c_int64_t, virtuals * virtuals - 1_c_int64_t
      e = task / virtuals
      q = modulo(task, virtuals)
      value = eri(tensor4_offset(occupied + a, occupied + b, &
                                 occupied + e, occupied + q, &
                                 orbitals, orbitals, orbitals))
      do m = 0_c_int64_t, occupied - 1_c_int64_t
        value = value - &
            t1(matrix_offset(m, b, virtuals)) * &
            eri(tensor4_offset(occupied + a, m, occupied + e, occupied + q, &
                               orbitals, orbitals, orbitals)) + &
            t1(matrix_offset(m, a, virtuals)) * &
            eri(tensor4_offset(occupied + b, m, occupied + e, occupied + q, &
                               orbitals, orbitals, orbitals))
      end do
      do m = 0_c_int64_t, occupied - 1_c_int64_t
        do n = 0_c_int64_t, occupied - 1_c_int64_t
          value = value + 0.25_c_double * &
              tau_ab(matrix_offset(m, n, occupied)) * &
              eri(tensor4_offset(m, n, occupied + e, occupied + q, &
                                 orbitals, orbitals, orbitals))
        end do
      end do
      wabef(matrix_offset(e, q, virtuals)) = value
    end do
    !$omp end parallel do
  end subroutine lccsdt_legacy_wabef_block_pair_scalars

end module lccsdt_legacy_intermediate_kernels
