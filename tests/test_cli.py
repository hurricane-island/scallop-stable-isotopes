from pathlib import Path
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

def test_isotopes_plot_tissue_scatter_lipids():
    """Test the plot tissue scatter lipid extraction command-line interface."""
    runner = CliRunner()
    output = figures / "lipid_extraction_wild_muscle_carbon_fractionation.png"
    already_exists = output.exists()
    last_modified_time = output.stat().st_mtime if already_exists else None
    result = runner.invoke(plot, [
        'tissue', 
        'scatter', 
        'lipids', 
        '--tissue-type', TissueType.MUSCLE.name,
        '--culture-method', CultureMethod.WILD.name,
        '--compare', Dimension.CARBON_FRACTIONATION.name,
    ])
    assert result.exit_code == 0
    if already_exists and last_modified_time is not None:
        new_modified_time = output.stat().st_mtime
        assert new_modified_time > last_modified_time
    else:
        assert output.exists()