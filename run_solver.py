from src.solvers.mathematical_constants_solver import MathematicalConstantsSolver

def main():
    solver = MathematicalConstantsSolver()
    target_address = "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH"
    public_key = "0456b3817434935db42afda0165de529b938cf67c7510168dbbe075f6b4da00f7769133b7c47ba4aba5eca84d4d7cdc644e231f5bb0adb7af34d1aec5c0891add9"
    difficulty = 64

    result = solver.solve(target_address, public_key, difficulty)
    if result:
        print(f"Solved private key: {hex(result)[2:].zfill(64)}")
    else:
        print("Failed to solve the puzzle")

if __name__ == "__main__":
    main()
