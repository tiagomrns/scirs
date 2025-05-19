use ndarray::array;
use scirs2_stats::gini_coefficient;

fn main() {
    // Example with income data - perfect equality
    let equal_data = array![100.0, 100.0, 100.0, 100.0, 100.0];
    let gini_equal = gini_coefficient(&equal_data.view()).unwrap();
    println!("Gini coefficient (perfect equality): {}", gini_equal);
    // Should be 0.0

    // Example with income data - perfect inequality (one person has everything)
    let unequal_data = array![0.0, 0.0, 0.0, 0.0, 100.0];
    let gini_unequal = gini_coefficient(&unequal_data.view()).unwrap();
    println!("Gini coefficient (perfect inequality): {}", gini_unequal);
    // Should be close to 0.8 (1 - 1/n, where n=5)

    // Example with real-world income data
    let income_data = array![
        20000.0, 25000.0, 30000.0, 35000.0, 40000.0, 45000.0, 50000.0, 60000.0, 80000.0, 150000.0
    ];
    let gini_income = gini_coefficient(&income_data.view()).unwrap();
    println!(
        "Gini coefficient (realistic income distribution): {}",
        gini_income
    );

    // Example with negative values (not a valid use case for Gini)
    let data_with_negative = array![10.0, -5.0, 20.0, 30.0, -10.0];
    match gini_coefficient(&data_with_negative.view()) {
        Ok(gini) => println!("Gini coefficient (with negative values): {}", gini),
        Err(e) => println!("Error: {}", e),
    }
}
