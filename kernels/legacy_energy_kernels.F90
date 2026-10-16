! Mixed-language CCSD(T) kernel boundary for CCSD-playground.
!
! The row-major linear offsets deliberately match localized_cc::Tensor.  This
! avoids hidden transposes at the C++/Fortran boundary.  The equations and
! prefactors are the spin-orbital conventions documented by this repository.
module lccsdt_legacy_energy_kernels
  use, intrinsic :: iso_c_binding, only: c_double, c_int, c_int64_t
  implicit none
  private
  public :: lccsdt_legacy_ccsd_energy, lccsdt_legacy_triples_energy

contains

  pure integer(c_int64_t) function matrix_offset(row, column, columns) result(offset)
    integer(c_int64_t), value :: row, column, columns
    offset = 1_c_int64_t + row * columns + column
  end function matrix_offset

  pure integer(c_int64_t) function tensor4_offset(i, j, a, b, dim_j, dim_a, dim_b) result(offset)
    integer(c_int64_t), value :: i, j, a, b, dim_j, dim_a, dim_b
    offset = 1_c_int64_t + (((i * dim_j + j) * dim_a + a) * dim_b + b)
  end function tensor4_offset

  subroutine lccsdt_legacy_ccsd_energy(rank, ranks, occupied, virtuals, fock, eri, t1, t2, energy, status) &
      bind(C, name="lccsdt_legacy_ccsd_energy")
    integer(c_int64_t), value, intent(in) :: rank, ranks, occupied, virtuals
    real(c_double), intent(in) :: fock(*), eri(*), t1(*), t2(*)
    real(c_double), intent(out) :: energy
    integer(c_int), intent(out) :: status
    integer(c_int64_t) :: i, j, a, b, orbitals, task
    real(c_double) :: integral

    energy = 0.0_c_double
    status = 0_c_int
    if (rank < 0_c_int64_t .or. ranks <= 0_c_int64_t .or. rank >= ranks .or. &
        occupied <= 0_c_int64_t .or. virtuals <= 0_c_int64_t) then
      status = 1_c_int
      return
    end if
    orbitals = occupied + virtuals

    !$omp parallel do default(none) &
    !$omp shared(rank, ranks, occupied, virtuals, orbitals, fock, t1) &
    !$omp private(i, a) reduction(+:energy)
    do task = rank, occupied * virtuals - 1_c_int64_t, ranks
      i = task / virtuals
      a = modulo(task, virtuals)
      energy = energy + &
          fock(matrix_offset(i, occupied + a, orbitals)) * &
          t1(matrix_offset(i, a, virtuals))
    end do
    !$omp end parallel do

    !$omp parallel do default(none) &
    !$omp shared(rank, ranks, occupied, virtuals, orbitals, eri, t1, t2) &
    !$omp private(i, j, a, b, integral) reduction(+:energy)
    do task = rank, occupied * occupied - 1_c_int64_t, ranks
      i = task / occupied
      j = modulo(task, occupied)
      do a = 0_c_int64_t, virtuals - 1_c_int64_t
        do b = 0_c_int64_t, virtuals - 1_c_int64_t
          integral = eri(tensor4_offset(i, j, occupied + a, occupied + b, &
                                         orbitals, orbitals, orbitals))
          energy = energy + 0.25_c_double * integral * &
              t2(tensor4_offset(i, j, a, b, occupied, virtuals, virtuals)) + &
              0.5_c_double * integral * t1(matrix_offset(i, a, virtuals)) * &
              t1(matrix_offset(j, b, virtuals))
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine lccsdt_legacy_ccsd_energy

  subroutine lccsdt_legacy_triples_energy(rank, ranks, count, left_moment, response, energy, status) &
      bind(C, name="lccsdt_legacy_triples_energy")
    integer(c_int64_t), value, intent(in) :: rank, ranks, count
    real(c_double), intent(in) :: left_moment(*), response(*)
    real(c_double), intent(out) :: energy
    integer(c_int), intent(out) :: status
    integer(c_int64_t) :: element

    energy = 0.0_c_double
    status = 0_c_int
    if (rank < 0_c_int64_t .or. ranks <= 0_c_int64_t .or. rank >= ranks .or. &
        count <= 0_c_int64_t) then
      status = 1_c_int
      return
    end if
    !$omp parallel do default(none) shared(rank, ranks, count, left_moment, response) reduction(+:energy)
    do element = rank + 1_c_int64_t, count, ranks
      energy = energy + left_moment(element) * response(element)
    end do
    !$omp end parallel do
    energy = energy / 36.0_c_double
  end subroutine lccsdt_legacy_triples_energy

end module lccsdt_legacy_energy_kernels
