"""
Unit tests for the isotopes CLI commands.
"""
from pathlib import Path
from typing import Any, Callable
from click.testing import CliRunner
from isotopes.describe import describe
from isotopes.plot import plot
from isotopes.options import TissueType, Dimension, figures, CultureMethod


def test_isotopes_describe_help():
    """Test the describe command-line interface."""
    runner = CliRunner()
    result = runner.invoke(describe, ['--help'])
    assert result.exit_code == 0
    assert 'Usage' in result.output

def test_isotopes_describe_tissue_anova():
    """Test basic ANOVA usage."""
    runner = CliRunner()
    result = runner.invoke(describe, [
        'tissue', 
        'anova', 
        '--tissue-type', 'muscle'
    ])
    assert result.exit_code == 0

def test_isotopes_describe_tissue_levenes():
    """Test basic Levene's test usage."""
    runner = CliRunner()
    result = runner.invoke(describe, [
        'tissue', 
        'levenes', 
        '--tissue-type', TissueType.MUSCLE.name,
        '--group-by', Dimension.GEAR.name,
        '--variable', Dimension.MOLAR_RATIO.name
    ])
    assert result.exit_code == 0

def test_isotopes_describe_tissue_outliers():
    """Test the describe tissue outliers command-line interface."""
    runner = CliRunner()
    result = runner.invoke(describe, [
        'tissue', 
        'outliers', 
        '--tissue-type', TissueType.MUSCLE.name,
        '--variable', Dimension.MOLAR_RATIO.name
    ])
    assert result.exit_code == 0

def test_isotopes_describe_tissue_by_gear_and_month():
    """Test the describe tissue by gear and month command-line interface."""
    runner = CliRunner()
    result = runner.invoke(describe, [
        'tissue', 
        'gear' 
    ])
    assert result.exit_code == 0

def test_isotopes_describe_environment_temperature():
    """Test the describe environment temperature command-line interface."""
    runner = CliRunner()
    result = runner.invoke(describe, [
        'environment', 
        'temperature', 
        '--threshold', '55.4',
    ])
    assert result.exit_code == 0

def existence_or_update(filepath: Path, fcn: Callable) -> Any:
    """Check if the output file exists or has been updated."""
    already_exists = filepath.exists()
    last_modified_time = filepath.stat().st_mtime if already_exists else None
    result = fcn()
    if result.exit_code != 0:
        print(result.output)
    assert result.exit_code == 0
    if already_exists and last_modified_time is not None:
        new_modified_time = filepath.stat().st_mtime
        assert new_modified_time > last_modified_time
    else:
        assert filepath.exists()
    return result

def test_isotopes_plot_tissue_scatter_lipids():
    """Test the plot tissue scatter lipid extraction command-line interface."""
    output = figures / "lipid_extraction_wild_muscle_carbon_fractionation.png"
    def _test_plot():
        runner = CliRunner()
        return runner.invoke(plot, [
            'tissue', 
            'scatter', 
            'lipids', 
            '--tissue-type', TissueType.MUSCLE.name,
            '--culture-method', CultureMethod.WILD.name,
            '--compare', Dimension.CARBON_FRACTIONATION.name,
        ])
    _ = existence_or_update(output, _test_plot)

def test_isotopes_plot_tissue_pairs():
    """Seaborn pairplot"""
    output = figures / "new-pairplot-all-tissue-gear.png"
    def _test_plot():
        runner = CliRunner()
        return runner.invoke(plot, [
            'tissue',
            'pairs',
        ])
    _ = existence_or_update(output, _test_plot)

def test_isotopes_plot_scatter_monthly_gear():
    """Something about tissue, not to be confused with the other one"""
    output = figures / "rawdata_scatter_monthly_gear.png"
    def _test_plot():
        runner = CliRunner()
        return runner.invoke(plot, [
            'tissue', 
            'monthly', 
            '--tissue-type', TissueType.MUSCLE.name,
        ])
    _ = existence_or_update(output, _test_plot)


def test_isotopes_plot_scatter_gear_monthly():
    """Something about tissue, not to be confused with the other one"""
    output = figures / "rawdata_scatter_gear_monthly.png"
    def _test_plot():
        runner = CliRunner()
        return runner.invoke(plot, [
            'tissue', 
            'scatter-gear-monthly', 
            '--tissue-type', TissueType.MUSCLE.name,
        ])
    _ = existence_or_update(output, _test_plot)

def test_isotopes_plot_box_var():
    """Test the plot tissue box var command-line interface."""
    output = figures / "isotopes_plot_box_var_molar_ratio.png"
    def _test_plot():
        runner = CliRunner()
        return runner.invoke(plot, [
            'tissue',
            'box',
            'var', 
            Dimension.MOLAR_RATIO.name,
        ])
    _ = existence_or_update(output, _test_plot)

def test_isotopes_plot_box_gsi():
    """Test the plot tissue box gsi command-line interface."""
    output = figures / "isotopes_plot_box_gsi.png"
    def _test_plot():
        runner = CliRunner()
        return runner.invoke(plot, [
            'tissue',
            'box',
            'gsi'
        ])
    _ = existence_or_update(output, _test_plot)

def test_isotopes_plot_tissue_histograms():
    """Test the plot tissue histograms command-line interface."""
    output = figures / "tissue_histograms_muscle.png"
    def _test_plot():
        runner = CliRunner()
        return runner.invoke(plot, [
            'tissue',
            'histograms',
            '--tissue-type', TissueType.MUSCLE.name
        ])
    _ = existence_or_update(output, _test_plot)

def test_isotopes_plot_famd_analysis_2d():
    """Test the plot famd analysis 2d command-line interface."""
    output = figures / "factor_analysis_of_mixed_data.png"
    def _test_plot():
        runner = CliRunner()
        return runner.invoke(plot, [
            'tissue',
            'scatter',
            'famd',
            '--tissue-type', TissueType.MUSCLE.name,
            '--convex-hulls'
        ])
    _ = existence_or_update(output, _test_plot)
